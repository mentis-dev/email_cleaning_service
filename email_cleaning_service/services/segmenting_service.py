import logging
import tensorflow as tf
import email_cleaning_service.data_model.data as data
import email_cleaning_service.data_model.pipelining as pipe
from email_cleaning_service.utils.data_manipulation import flatten_list, batch_list
from email_cleaning_service.config import DEVICE


def segment(dataset: data.EmailDataset, pipeline: pipe.PipelineModel) -> None:
    """Used to segment all EmailThread objects in the dataset
    pipeline must be a valid PipelineModel object
    """
    cat_preds, frag_preds = [], []
    logging.info("Segmenting emails...")
    for batch in dataset.get_tf_dataset():
        # Run pipeline on batch
        with tf.device(DEVICE):  # type: ignore
            pred = pipeline(batch)  # type: ignore

        # Separate results
        cat_preds.append(tf.argmax(pred[:, :, :7], axis=-1))  # type: ignore
        frag_preds.append(pred[:, :, -1])  # type: ignore

    # Combine results into single tensor
    cat_pred = tf.concat(cat_preds, axis=0)  # type: ignore
    frag_pred = tf.concat(frag_preds, axis=0)  # type: ignore

    # Get dictionary to later combine sequences into threads
    seq_order = dataset.seq_order
    concat_cat_pred = {seq: [] for seq in seq_order}
    concat_frag_pred = {seq: [] for seq in seq_order}
    for i, seq in enumerate(seq_order):
        concat_cat_pred[seq].append(cat_pred[i])
        concat_frag_pred[seq].append(frag_pred[i])

    # Segment each thread in the batch
    for key in concat_cat_pred.keys():
        cat_pred, frag_pred = flatten_list(concat_cat_pred[key]), flatten_list(
            concat_frag_pred[key]
        )
        dataset.threads[key].segment(cat_pred, frag_pred)
    dataset.is_labeled = True
    logging.info("Segmentation complete")
