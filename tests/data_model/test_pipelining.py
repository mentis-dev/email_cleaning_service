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
    
    text = "Hello, my name is John Doe! My phone number is +33781624352. My email is paultfetrer@hello.fr"
    for extracted, needed in zip(extractor(tf.constant([text])).numpy()[0][0], [1, 0, 1, 0, 0, 0, 1, 5, 0]):
        print(extracted, needed)
        assert int(extracted) == needed

def test_BareEncoder():
    
    encoder = pipelining.BareEncoder("sentence-transformers/all-MiniLM-L6-v2")

def test_EncoderModel():
    
    encoder = pipelining.EncoderModel.from_hugg("sentence-transformers/all-MiniLM-L6-v2")

    assert encoder(tf.constant(["Hello, my name is John Doe!"])).get_shape().as_list() == [1, 384]

def test_FeatureCreator():
    pass

def test_ClassifierModel():
    pass

def test_Pipeline():
    pass
