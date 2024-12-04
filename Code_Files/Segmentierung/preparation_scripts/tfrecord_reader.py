import tensorflow as tf


def list_keys_in_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    for raw_record in dataset.take(1):  # Take the first record to inspect
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


list_keys_in_tfrecord(r'.tfrecord_file')
