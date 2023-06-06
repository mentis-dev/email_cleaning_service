import sys
import os
import logging

if "../" not in sys.path:
    sys.path.append("../")

import mlflow
import tensorflow as tf
from keras.layers import Dense, Concatenate, Bidirectional, GRU, Dropout


mlflow.set_tracking_uri(sys.argv[1])
mlflow.set_experiment(experiment_name="EC_classifier_training")

logging.info("Creating default model")
i = tf.keras.Input(shape=(64, 393))

i = tf.keras.layers.Reshape((64, 393, 1), input_shape=(64, 393))(i)

first_half = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 9)))(i)
first_half = tf.keras.layers.Reshape((64, 384), input_shape=(64, 384, 1))(first_half)

second_half = tf.keras.layers.Cropping2D(cropping=((0, 0), (384, 0)))(i)
second_half = tf.keras.layers.Reshape((64, 9), input_shape=(64, 9, 1))(second_half)

second_half = Dense(128, activation="relu")(second_half)

x = Concatenate(axis=-1)([first_half, second_half])

x = Bidirectional(GRU(256, return_sequences=True))(x)
x = Dropout(0.25)(x)
x = Bidirectional(GRU(128, return_sequences=True))(x)
x = Dropout(0.25)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
section_x = Dense(7, activation="softmax")(x)
fragment_x = Dense(1, activation="sigmoid")(x)
x = Concatenate(axis=-1)([section_x, fragment_x])

clf = tf.keras.Model(inputs=i, outputs=x)

logging.info("logging default model")
with mlflow.start_run(run_name="default"):
    mlflow.log_params(
        {
            "encoder_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "encoder_dim": 384,
            "features": [
                "phone_number",
                "url",
                "punctuation",
                "horizontal_separator",
                "hashtag",
                "pipe",
                "email",
                "capitalized",
                "full_caps",
            ],
        }
    )

    mlflow.tensorflow.log_model(clf, "classifier")

logging.info("Creating optimized model")
clf.load_weights(
    "./temp/base_multi_miniLM_classifier_optimized/multi_miniLM_classifier.h5"
)

logging.info("logging optimized model")
with mlflow.start_run(run_name="optimized"):
    mlflow.log_params(
        {
            "encoder_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "encoder_dim": 384,
            "features": [
                "phone_number",
                "url",
                "punctuation",
                "horizontal_separator",
                "hashtag",
                "pipe",
                "email",
                "capitalized",
                "full_caps",
            ],
        }
    )

    mlflow.tensorflow.log_model(clf, "classifier")

logging.info("Done")
