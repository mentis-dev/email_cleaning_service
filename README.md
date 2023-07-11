# email_cleaning_service

This is an email segmenting service which takes a list of emails as input and returns the the header, body and signature of each message in the email

## Getting Started

The project is published on Pypi and can be installed using the following command

```py
pip install email-cleaning-service
```

## Usage

The package can be used as follows
```py
from email_cleaning_service.control import EmailCleaner

email_cleaner = EmailCleaner(tracking_uri, storage_uri)
```

Usage revolves around the emailCleaner classwhich is the preferred interface for the package. The class takes two arguments, the tracking_uri and the storage_uri. The tracking_uri is the uri of the MLflow tracking server and the storage_uri is the uri of the storage server (can be a path to a local folder).

BaseModel classes exist to simplify interactions with the class. The most important of these is the PipelineSpecs class which is used to define the pipeline to be used for cleaning the emails.

```py
from email_cleaning_service.utils.request_classes import PipelineSpecs

pipeline_specs = PipelineSpecs(
    classifier_origin="mlflow", # or "h5" or "config
    classifier_id="a1f66311816e417cb94db7c2457b25d1",
    encoder_origin="hugg", # or "mlflow"
    encoder_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encoder_dim=384,
    features=[
            "phone_number",
            "url",
            "punctuation",
            "horizontal_separator",
            "hashtag",
            "pipe",
            "email",
            "capitalized",
            "full_caps"
        ]      # Can be any combination of the above features
)
```

**The above pipeline is the recommended one for multilingual email segmenting for now.**

The pipeline contains 3 main elements:
* The embedding model: This is the model used to embed the emails into a vector space. The model can be either a huggingface model or an MLflow model. If the model is from hugging face, specify encoder_origin as "hugg" and the encoder_id as the model name on the platform. If the model is from MLflow, specify encoder_origin as "mlflow" and the encoder_id as the id of the run you want to use the model from on the MLFlow server.
* The extracted features: a list of regex features that are concatenated to the embedding of each sentence
* The classifier: This is the model used to realise the final classification and the separation of a thread in multiple messages. The model can come from mlflow in which case the run_id must be specified or from a h5 file in which case the path to the file must be specified.



The pipeline can then be used as follows:
```py
email_list = [
    "This is a test email. I am testing the email cleaner.\nYours truly, Paul",
    "Hello team!\nThis is another test email with two lines.\n I am testing the email cleaner",
    "Bonjour!\nCeci est un autre email\n\nAu revoir!\nPaul",
]

email_cleaner.segment(email_list, pipeline_specs)
```

The output should look like this:

```py
{'threads': [{'source': 'This is a test email. I am testing the email cleaner.\nYours truly, Paul\n0781759532',
   'messages': [{'full': 'This is a test email. I am testing the email cleaner.\nYours truly, Paul\n0781759532',
     'header': '',
     'disclaimer': '',
     'greetings': '',
     'body': 'This is a test email. I am testing the email cleaner.\nYours truly, Paul\n0781759532',
     'signature': '',
     'caution': ''}]},
  {'source': 'Hello team!\nThis is another test email with two lines.\n I am testing the email cleaner.',
   'messages': [{'full': 'Hello team!\nThis is another test email with two lines.\n I am testing the email cleaner.',
     'header': '',
     'disclaimer': '',
     'greetings': '',
     'body': 'Hello team!',
     'signature': 'This is another test email with two lines.\n I am testing the email cleaner.',
     'caution': ''}]},
  {'source': 'Bonjour!\nCeci est un autre email\n\nAu revoir!\nPaul',
   'messages': [{'full': 'Bonjour!\nCeci est un autre email\nAu revoir!\nPaul',
     'header': '',
     'disclaimer': '',
     'greetings': '',
     'body': 'Bonjour!\nCeci est un autre email\nAu revoir!',
     'signature': 'Paul',
     'caution': ''}]}]}
```

## Model Training

This package also includes support for training pipelines. You can either train (fine-tune) an encoder model or a classifier model. A note-worthy difference between the 2 is that encoders are trained with a single line of the email as input while classifiers are trained with a sequence of 64 lines as input.

The csv files used for training must use be contain lines from emails and have the following columns:
* Email: a unique Id for each email used to group email lines together
* Text: the text of the email line
* Section: the section of the email line (disclaimer, header, greetings, body, signature and caution represented as 1 thru 6 respectively)
* FragmentChange: a boolean (0 or 1) indicating whether the line is a fragment change or not

### Training an Encoder

To train an encoder, use the EncoderSpecs class and the RunSpecs class as follows:

```py
from email_cleaning_service.utils.request_classes import EncoderSpecs, RunSpecs

dataset = RunSpecs(
    run_name="demo_encoder_test_run_2",
    csv_train="./train_multi_71.csv",
    csv_test="./test_multi_48.csv",
    batch_size=4,
    metrics=[],
    lr=0.0001,
    epochs=1,
)

encoder_specs = EncoderSpecs(
    origin="mlflow",
    encoder="14a633237e734575ad7f8eac9bd0319e"
)

email_cleaner.train_encoder(dataset, encoder_specs)
```

The EncoderSpecs class takes two arguments, origin and encoder which are the same as in PipelineSpecs. 
The RunSpecs define how you want to train the model. The arguments are:
* run_name: The name of the run on the MLflow server
* csv_train: The path to the csv file containing the training data
* csv_test: The path to the csv file containing the test data
* batch_size: The batch size to use for training
* metrics: A list of metrics to track during training. The metrics must be defined in the metrics.py file in utils
* lr: The learning rate to use for training
* epochs: The number of epochs to train for

### Training a Classifier

To train a classifier, an entire pipeline must be defined. This is done using the PipelineSpecs class as follows:

```py
dataset = RunSpecs(
    run_name="with_fine_tuned_encoder",
    batch_size=4,
    csv_train="./train_multi.csv",
    csv_test="./test_multi.csv",
    metrics=["seq_f1", "frag_f1"],
    lr=0.007,
    epochs=3,
)

pipeline_specs = PipelineSpecs(
    classifier_origin="h5",
    classifier_id="./temp/base_multi_miniLM_classifier_optimized/multi_miniLM_classifier.h5",
    encoder_origin="mlflow",
    encoder_id="316fb5040b0a4353ade2e967290944ff",
    encoder_dim=384,
    features=[
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

email_cleaner.train_classifier(dataset, pipeline_specs)
```

## Package Structure

![package_architecture](./assets/package_architecture.png)

## Maintaining the Package



## Known Issues








