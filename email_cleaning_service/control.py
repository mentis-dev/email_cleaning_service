import logging
import email_cleaning_service.data_model.data as data
import email_cleaning_service.data_model.pipelining as pipe
import email_cleaning_service.utils.request_classes as rq
from typing import List
import email_cleaning_service.services.segmenting_service as seg
import email_cleaning_service.services.training_service as train
import mlflow


class EmailCleaner:
    """Controller class for the API. Used to control the pipeline and dataset objects."""

    def __init__(self, tracking_uri: str = "https://mentis.io/mlflow/"):
        mlflow.set_tracking_uri(tracking_uri)
        logging.info("Controller initialized")

    def segment(
        self, thread_list: List[str], pipeline_specs: rq.PipelineSpecs
    ) -> data.EmailDataset:
        """Used to segment all EmailThread objects in the dataset
        pipeline must be a valid PipelineModel object
        """
        dataset = data.EmailDataset(thread_list)
        if pipeline_specs.origin == "hugg":
            pipeline = pipe.PipelineModel.from_hugg(pipeline_specs)
        elif pipeline_specs.origin == "mlflow":
            pipeline = pipe.PipelineModel.from_mlflow(pipeline_specs)
        else:
            raise ValueError("Invalid pipeline origin")
        seg.segment(dataset, pipeline)
        return dataset

    def train_classifier(
        self, run_specs: rq.RunSpecs, pipeline_specs: rq.PipelineSpecs
    ):
        """Used to train the classifier on the dataset
        pipeline must be a valid PipelineModel object
        """
        dataset = data.EmailDataset.from_csv(run_specs.csv_path)
        if pipeline_specs.origin == "hugg":
            pipeline = pipe.PipelineModel.from_hugg(pipeline_specs)
        elif pipeline_specs.origin == "mlflow":
            pipeline = pipe.PipelineModel.from_mlflow(pipeline_specs)
        else:
            raise ValueError("Invalid pipeline origin")
        train.train_classifier(run_specs, dataset, pipeline)

    def train_encoder(self, run_specs: rq.RunSpecs, encoder_specs: rq.EncoderSpecs):
        """Used to train the encoder on the dataset
        pipeline must be a valid PipelineModel object
        """
        dataset = data.EmailLineDataset.from_csv(run_specs.csv_path)
        if encoder_specs.origin == "hugg":
            encoder = pipe.EncoderModel.from_hugg(encoder_specs.encoder)
        elif encoder_specs.origin == "mlflow":
            encoder = pipe.EncoderModel.from_mlflow(encoder_specs.encoder)
        else:
            raise ValueError("Invalid encoder origin")
        train.train_encoder(run_specs, dataset, encoder)  # type: ignore



