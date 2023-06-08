from email_cleaning_service.control import EmailCleaner
from email_cleaning_service.utils.request_classes import PipelineSpecs
import pytest

def test_segmenting_service():
    tracking_uri = "https://mentis.io/mlflow/"

    emailCleaner = EmailCleaner()

    # Example of how to segment a dataset

    thread_list = [
        "This is a test email. I am testing the email cleaner.",
        "This is another test email with two lines.\n I am testing the email cleaner.",
    ]

    pipeline_specs = PipelineSpecs(
        origin="hugg",
        classifier_id="a1f66311816e417cb94db7c2457b25d1"
    )

    dataset = emailCleaner.segment(thread_list, pipeline_specs)
    assert len(dataset.threads) == 2