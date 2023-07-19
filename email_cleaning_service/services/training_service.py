import logging
import tensorflow as tf
import email_cleaning_service.data_model.data as data
import email_cleaning_service.data_model.pipelining as pipe
import email_cleaning_service.utils.request_classes as rq
import matplotlib.pyplot as plt
from email_cleaning_service.utils.metrics import METRICS, multifactor_loss
from transformers import AutoTokenizer
import mlflow
from email_cleaning_service.config import DEVICE                                                              


def train_classifier(run_specs: rq.RunSpecs, train_dataset: data.EmailDataset, test_dataset: data.EmailDataset, pipeline: pipe.PipelineModel) -> None:
    """Used to train the classifier on the dataset
    pipeline must be a valid PipelineModel object
    """

    mlflow.set_experiment("EC_classifier_training")

    # get the generator which will feed the classifier training
    def _get_generator(tf_dataset: tf.data.Dataset, feature_creator: pipe.FeatureCreator):
        def gen():
            for text, label in tf_dataset:
                yield feature_creator(text) , label
        return gen

    logging.info("Training classifier...")
    train_tf_dataset = train_dataset.set_batch_size(run_specs.batch_size).get_tf_dataset()
    test_tf_dataset = test_dataset.set_batch_size(run_specs.batch_size).get_tf_dataset()
    feature_creator = pipeline.encoder
    classifier = pipeline.classifier
    train_feature_generator = _get_generator(train_tf_dataset, feature_creator)
    test_feature_generator = _get_generator(test_tf_dataset, feature_creator)


    # Defining run parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=run_specs.lr)
    metrics_fn = {metric: METRICS[metric] for metric in run_specs.metrics}
    classifier.classifier.compile(optimizer=optimizer, loss=multifactor_loss(), metrics=metrics_fn.values()) # type: ignore

    # Starting run
    with mlflow.start_run(run_name=run_specs.run_name):
        mlflow.log_params({
            "epochs": run_specs.epochs,
            "batch_size": run_specs.batch_size,
            "optimizer": "adam",
            "lr": run_specs.lr,
            "loss": "multifactor_loss",
            "metrics": run_specs.metrics,
            "encoder_id": pipeline.encoder_id,
            "encoder_dim": pipeline.encoder_dim,
            "features": pipeline.features,
        })

        with tf.device(DEVICE): # type: ignore
            fit_history = classifier.classifier.fit(train_feature_generator(), epochs=run_specs.epochs) # type: ignore
            scores = classifier.classifier.evaluate(test_feature_generator(), verbose=1) # type: ignore
        
        mlflow.log_metrics({
            metric_name: fit_history.history[metric_fn.__name__][-1] for metric_name, metric_fn in metrics_fn.items()
        })
        mlflow.log_metrics({
            f"val_{metric_name}": score for metric_name, score in zip(classifier.classifier.metrics_names, scores)
        })

        mlflow.tensorflow.log_model(classifier.classifier, "classifier")
    logging.info("Training complete")


def train_encoder(run_specs: rq.RunSpecs, train_dataset: data.EmailLineDataset, test_dataset: data.EmailLineDataset, encoder: pipe.EncoderModel, storage_uri: str) -> None:
    """Used to train the encoder on the dataset
    pipeline must be a valid PipelineModel object
    """

    mlflow.set_experiment("EC_encoder_training")

    def _get_generator(tf_dataset: tf.data.Dataset, tokenizer: AutoTokenizer):
        def gen():
            for text, label in tf_dataset:
                lines = [str(line) for line in text.numpy()]
                tokens = tokenizer(lines, padding=True, truncation=True, return_tensors='tf') # type: ignore
                yield { "input_ids": tf.convert_to_tensor(tokens["input_ids"], dtype=tf.int32), "token_type_ids": tf.convert_to_tensor(tokens["token_type_ids"], dtype=tf.int32), "attention_mask": tf.convert_to_tensor(tokens["attention_mask"], dtype=tf.int32)}, label
        return gen
    
    def _get_dataset(tf_dataset: tf.data.Dataset, tokenizer: AutoTokenizer):
        return tf.data.Dataset.from_generator(
            _get_generator(tf_dataset, tokenizer), 
            output_types=({ "input_ids": tf.int32, "token_type_ids": tf.int32, "attention_mask": tf.int32 }, tf.int32)
        )

    logging.info("Training encoder...")
    train_tf_dataset = train_dataset.set_batch_size(batch_size=run_specs.batch_size).get_tf_dataset()
    test_tf_dataset = test_dataset.set_batch_size(batch_size=run_specs.batch_size).get_tf_dataset()
    tokenizer = encoder.tokenizer
    encoder_model = encoder.model

    train_feature_generator = _get_dataset(train_tf_dataset, tokenizer)
    test_feature_generator = _get_dataset(test_tf_dataset, tokenizer)

    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    token_type_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name="token_type_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    
    x = encoder_model([input_ids, token_type_ids, attention_mask]) # type: ignore
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(7, activation='softmax')(x)
    clf = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=run_specs.lr)
    clf.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

    with mlflow.start_run(run_name=run_specs.run_name) as run:

        mlflow.log_params({
            "epochs": run_specs.epochs,
            "batch_size": run_specs.batch_size,
            "optimizer": "adam",
            "lr": run_specs.lr,
            "loss": "sparse_categorical_crossentropy"
        })

        with tf.device(DEVICE): # type: ignore
            fit_history = clf.fit(train_feature_generator, epochs=run_specs.epochs)
            scores = clf.evaluate(test_feature_generator)
        
        mlflow.log_metrics({"loss": fit_history.history["loss"][-1],
                            "val_loss": scores})

        logging.info("Saving model in temp files")
        save_directory = f"{storage_uri.strip('/')}/{run.info.run_id}"
        encoder_model.model.save_pretrained(save_directory + "/encoder") # type: ignore
        tokenizer.save_pretrained(save_directory + "/tokenizer") # type: ignore
        logging.info("Done")
    logging.info("Training complete")

def evaluate(test_dataset: data.EmailDataset, pipeline: pipe.PipelineModel) -> None:
    """Used to evaluate the pipeline on the dataset
    pipeline must be a valid PipelineModel object
    """

    # get the generator which will feed the classifier training
    def _get_generator(tf_dataset: tf.data.Dataset, feature_creator: pipe.FeatureCreator):
        def gen():
            for text, label in tf_dataset:
                yield feature_creator(text) , label
        return gen

    logging.info("Evaluating pipeline...")
    test_tf_dataset = test_dataset.set_batch_size(batch_size=4).get_tf_dataset()
    feature_creator = pipeline.encoder
    classifier = pipeline.classifier

    test_feature_generator = _get_generator(test_tf_dataset, feature_creator)
    classifier.classifier.compile(optimizer="adam", loss=multifactor_loss(), metrics=METRICS.values())

    with tf.device(DEVICE): # type: ignore
        scores = classifier.classifier.evaluate(test_feature_generator(), verbose=1) # type: ignore
    
    return scores





    
    