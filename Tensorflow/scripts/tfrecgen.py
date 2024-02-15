import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(feature1, feature2, feature3):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'feature1': _bytes_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _bytes_feature(feature3),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def generate_tfrecords(data, filename):
    with tf.io.tfRecordWriter(filename) as writer:
        for entry in data:
            # Assuming your data is in the format of (feature1, feature2, feature3)
            feature1, feature2, feature3 = entry

            example = serialize_example(
                feature1=tf.io.serialize_tensor(feature1),
                feature2=tf.io.serialize_tensor(feature2),
                feature3=tf.io.serialize_tensor(feature3)
            )
            writer.write(example)

# Example usage:
# Assuming your data is a list of tuples where each tuple contains the features
data = [
    (b'image1.jpg', b'label1', b'data1'),
    (b'image2.jpg', b'label2', b'data2'),
    # Add more data entries as needed
]

generate_tfrecords(data, 'data.tfrecords')
