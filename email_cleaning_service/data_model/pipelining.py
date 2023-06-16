import json
import logging
import os
import re
import tensorflow as tf
from typing import List, Generator, Any, Optional, Union
import email_cleaning_service.utils.request_classes as rq
from email_cleaning_service.utils.data_manipulation import parse_str_to_list
from keras.layers import Bidirectional, GRU, Dropout, Dense, Concatenate
from transformers import TFAutoModel, AutoTokenizer
import mlflow
from mlflow.models.signature import ModelSignature
import shutil

from email_cleaning_service.config import FEATURE_REGEX


class ExtractorModel:
    """Extracts features from text.
    Give a features_list with the features you want to extract from the following list:
    - phone_number
    - url
    - punctuation
    - horizontal_separator
    - hashtag
    - pipe
    - email
    - capitalized
    - full_caps
    """

    def __init__(self, features_list: List[str]) -> None:
        self.regex_list = [FEATURE_REGEX[feature] for feature in features_list]

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        feats = []
        for sequence in inputs.numpy():  # type: ignore
            feats.append([])
            for line in sequence:
                feats[-1].append(
                    [
                        self.extract_feature(line.decode("utf-8"), regex)
                        for regex in self.regex_list
                    ]
                )
        return tf.convert_to_tensor(feats, dtype=tf.float32)

    @staticmethod
    def extract_feature(text: str, regex: str) -> int:
        return len(re.findall(regex, text))


class BareEncoder(tf.keras.layers.Layer):
    model: TFAutoModel

    def __init__(self, model_name_or_path, **kwargs):
        super(BareEncoder, self).__init__()
        # loads transformers model
        self.model = TFAutoModel.from_pretrained(model_name_or_path, **kwargs)

    def call(self, inputs, normalize=True):
        # runs model on inputs
        model_output = self.model(inputs)  # type: ignore
        # Perform pooling. In this case, mean pooling.
        try:
            mask = inputs["attention_mask"]
        except TypeError:
            mask = inputs[2]
        embeddings = self.mean_pooling(model_output, mask)
        # normalizes the embeddings if wanted
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(
                tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)
            ),
            tf.float32,
        )
        return tf.math.reduce_sum(
            token_embeddings * input_mask_expanded, axis=1
        ) / tf.clip_by_value(
            tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max
        )

    def normalize(self, embeddings):
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)  # type: ignore
        return embeddings


class EncoderModel:
    """Encodes text into embeddings. Uses transformers models.
    specify model_name_or_path with the name of the model you want to use from hugging face.
    """

    tokenizer: AutoTokenizer
    model: BareEncoder

    def __init__(self, tokenizer_path: str, encoder_path: str, **kwargs) -> None:
        # loads transformers model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # type: ignore
        self.model = BareEncoder(encoder_path, **kwargs)  # type: ignore

    @classmethod
    def from_mlflow(cls, run_id: str) -> "EncoderModel":
        artifact_path = f"./temp/{run_id}"
        logging.info(f"Loading model from {artifact_path}")
        obj = cls(artifact_path + "/tokenizer", artifact_path + "/encoder")
        logging.info("Done")
        return obj

    @classmethod
    def from_hugg(cls, encoder_id: str) -> "EncoderModel":
        return cls(encoder_id, encoder_id)
    
    @classmethod
    def from_specs(cls, specs: str, **kwargs) -> "EncoderModel":
        if specs.origin == "mlflow":
            return cls.from_mlflow(specs.encoder)
        elif specs.origin == "hugg":
            return cls.from_hugg(specs.encoder)
        else:
            raise ValueError(f"Unknown origin {specs.origin}")

    def __call__(self, inputs: tf.Tensor, normalize: bool = True) -> tf.Tensor:
        tokenized = [
            self.tokenizer(
                [line.decode("utf-8") for line in lines],
                padding=True,
                truncation=True,
                return_tensors="tf",
            )  # type: ignore
            for lines in inputs.numpy()  # type: ignore
        ]
        return tf.convert_to_tensor(
            [self.model(token) for token in tokenized], dtype=tf.float32  # type: ignore
        )


class FeatureCreator:
    """Creates features from text. combines the encoder and the extractor.
    Give the name of the encoder model you want to use and the list of features you want to extract.
    Refer to the documentation of EncoderModel and ExtractorModel for more information."""

    encoder: EncoderModel
    extractor: ExtractorModel

    @classmethod
    def from_mlflow(cls, run_id: str, features_list: List[str]) -> "FeatureCreator":
        obj = cls()
        obj.encoder = EncoderModel.from_mlflow(run_id)
        obj.extractor = ExtractorModel(features_list)
        return obj

    @classmethod
    def from_hugg(cls, encoder_id: str, features_list: List[str]) -> "FeatureCreator":
        obj = cls()
        obj.encoder = EncoderModel.from_hugg(encoder_id)
        obj.extractor = ExtractorModel(features_list)
        return obj

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(inputs)
        extracted = self.extractor(inputs)
        return tf.concat([encoded, extracted], axis=-1)  # type: ignore


class ClassifierModel:
    """Classifies sequences of embeddings. Uses a RNN model.
    Specify classifier_name with the name of the model you want to use from the weights folder."""

    classifier = Union[tf.keras.Model, tf.keras.Sequential]

    @classmethod
    def from_config(cls, config: str) -> "ClassifierModel":
        obj = cls()
        obj.classifier = tf.keras.models.model_from_config(config)
        return obj

    @classmethod
    def from_mlflow(cls, classifier_id: str) -> "ClassifierModel":
        obj = cls()
        obj.classifier = mlflow.tensorflow.load_model(
            f"runs:/{classifier_id}/classifier", keras_model_kwargs={"compile": False}
        )
        return obj

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.classifier(inputs)  # type: ignore


class PipelineModel:
    """Combines the encoder, the extractor and the classifier.
    Specify model_name with the name of the model you want to use from the MODELS dictionary (or models.json file)."""

    encoder: FeatureCreator
    classifier: ClassifierModel
    encoder_id: str
    encoder_dim: int
    features: List[str]

    @classmethod
    def from_mlflow(cls, mlflow_specs: rq.PipelineSpecs) -> "PipelineModel":
        obj = cls()
        classifier_run = mlflow.tracking.MlflowClient().get_run(
            mlflow_specs.classifier_id
        )
        if not mlflow_specs.encoder_id:
            mlflow_specs.encoder_id = classifier_run.data.params["encoder_id"]
        obj.encoder_id = mlflow_specs.encoder_id
        if not mlflow_specs.features:
            mlflow_specs.features = parse_str_to_list(
                classifier_run.data.params["features"]
            )
        obj.features = mlflow_specs.features
        if not mlflow_specs.encoder_dim:
            mlflow_specs.encoder_dim = classifier_run.data.params["encoder_dim"]
        obj.encoder_dim = mlflow_specs.encoder_dim
        obj.encoder = FeatureCreator.from_mlflow(
            mlflow_specs.encoder_id, mlflow_specs.features
        )
        obj.classifier = ClassifierModel.from_mlflow(mlflow_specs.classifier_id)
        return obj

    @classmethod
    def from_hugg(cls, hugg_specs: rq.PipelineSpecs) -> "PipelineModel":
        obj = cls()
        classifier_run = mlflow.tracking.MlflowClient().get_run(
            hugg_specs.classifier_id
        )
        if not hugg_specs.encoder_id:
            hugg_specs.encoder_id = classifier_run.data.params["encoder_id"]
        obj.encoder_id = hugg_specs.encoder_id
        if not hugg_specs.features:
            hugg_specs.features = parse_str_to_list(
                classifier_run.data.params["features"]
            )
        obj.features = hugg_specs.features
        if not hugg_specs.encoder_dim:
            hugg_specs.encoder_dim = classifier_run.data.params["encoder_dim"]
        obj.encoder_dim = hugg_specs.encoder_dim
        obj.encoder = FeatureCreator.from_hugg(
            hugg_specs.encoder_id, hugg_specs.features
        )
        obj.classifier = ClassifierModel.from_mlflow(hugg_specs.classifier_id)
        return obj
    
    @classmethod
    def from_specs(cls, specs: rq.PipelineSpecs) -> "PipelineModel":
        if specs.classifier_origin == "mlflow" and not specs.encoder_origin:
            return cls.from_mlflow(specs)
        
        obj = cls()
        if specs.classifier_origin == "mlflow":
            obj.classifier = ClassifierModel.from_mlflow(specs.classifier_id)
        elif specs.classifier_origin == "specs":
            obj.classifier = ClassifierModel.from_config(specs.classifier_id)
        
        if specs.encoder_origin == "mlflow":
            obj.encoder = FeatureCreator.from_mlflow(specs.encoder_id, specs.features)
        elif specs.encoder_origin == "hugg":
            obj.encoder = FeatureCreator.from_hugg(specs.encoder_id, specs.features)
        
        return obj
        
        

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        result = self.classifier(self.encoder(inputs))
        return result
