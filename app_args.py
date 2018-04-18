import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 15, "The number of epochs; default 15")
flags.DEFINE_integer('batch_size', 64, "The batch size; default 64")
flags.DEFINE_float('learning_rate', 0.0001, "The learning rate; default 0.0001")
flags.DEFINE_string('loss', 'mse', 'The loss function; default "mse"')
flags.DEFINE_float('dropout', 0.2, 'The dropout probabilty; default 0.2')
flags.DEFINE_string('activation', 'relu', 'The activation function; default "relu"')
flags.DEFINE_float('batch_norm', 0.0, 'The batch norm momentum, if 0 batch norm is not applied; default 0')