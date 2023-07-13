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

    tracking_uri: str
    storage_uri: str

    def __init__(self, tracking_uri:str, storage_uri:str):
        self.tracking_uri = tracking_uri
        self.storage_uri = storage_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        logging.info("Controller initialized")

    def segment(
        self, thread_list: List[str], pipeline_specs: rq.PipelineSpecs
    ) -> data.EmailDataset:
        """Used to segment all EmailThread objects in the dataset
        pipeline must be a valid PipelineModel object
        """
        dataset = data.EmailDataset(thread_list)
        pipeline = pipe.PipelineModel.from_specs(pipeline_specs)
        seg.segment(dataset, pipeline)
        return dataset

    def train_classifier(
        self, run_specs: rq.RunSpecs, pipeline_specs: rq.PipelineSpecs
    ):
        """Used to train the classifier on the dataset
        pipeline must be a valid PipelineModel object
        """
        train_dataset = data.EmailDataset.from_csv(run_specs.csv_train)
        test_dataset = data.EmailDataset.from_csv(run_specs.csv_test)
        pipeline = pipe.PipelineModel.from_specs(pipeline_specs)
        train.train_classifier(run_specs, train_dataset, test_dataset, pipeline)

    def train_encoder(self, run_specs: rq.RunSpecs, encoder_specs: rq.EncoderSpecs):
        """Used to train the encoder on the dataset
        pipeline must be a valid PipelineModel object
        """
        train_dataset = data.EmailLineDataset.from_csv(run_specs.csv_train)
        test_dataset = data.EmailLineDataset.from_csv(run_specs.csv_test)
        encoder = pipe.EncoderModel.from_specs(encoder_specs)
        train.train_encoder(run_specs, train_dataset, test_dataset, encoder, self.storage_uri)  # type: ignore

    def evaluate(self, csv_test: str, pipeline_specs: rq.PipelineSpecs):
        """Used to evaluate the classifier on the dataset
        pipeline must be a valid PipelineModel object
        """
        test_dataset = data.EmailDataset.from_csv(csv_test)
        pipeline = pipe.PipelineModel.from_specs(pipeline_specs)
        scores = train.evaluate(test_dataset, pipeline)
        return scores



