import email_cleaning_service.data_model.pipelining as pipelining
import tensorflow as tf

def test_extractor_model():
    # Test that the extractor model is correctly loaded
    extractor = pipelining.ExtractorModel(
        features_list=[
            "phone_number",
            "url",
            "punctuation",
            "horizontal_separator",
            "hashtag",
            "pipe",
            "email",
            "capitalized",
            "full_caps"
        ]
    )
    

