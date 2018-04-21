import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 40, "The number of epochs; default 15")
flags.DEFINE_integer('batch_size', 64, "The batch size; default 64")
flags.DEFINE_float('learning_rate', 0.0001, "The learning rate; default 0.0001")
flags.DEFINE_string('loss', 'mse', 'The loss function; default "mse"')
flags.DEFINE_float('dropout', 0.2, 'The dropout probabilty; default 0.2')
flags.DEFINE_string('activation', 'relu', 'The activation function; default "relu"')
flags.DEFINE_float('batch_norm', 0.0, 'The batch norm momentum, if 0 batch norm is not applied; default 0')
flags.DEFINE_float('angle_correction', 0.15, 'The correction angle to add to right and left camera images')
flags.DEFINE_float('mirror_min_angle', 0.0, 'The min steering angle needed in order to discriminate if an image should be mirrored or not')
flags.DEFINE_float('normalize_factor', 1.5, 'The factor applied for normalization when dropping values over the mean')
flags.DEFINE_float('normalize_bins', 100, 'The number of bins to divide the data when normalizing')
flags.DEFINE_boolean('regenerate', False, 'True if the dataset should be regenerated from the log file')
flags.DEFINE_boolean('preprocess', True, 'True if the generator should preprocess all the images in memory')
flags.DEFINE_boolean('clahe', False, 'True if histogram equalization should be applied during preprocessing')
flags.DEFINE_boolean('blur', True, 'If blurring should be applied during preprocessing')
flags.DEFINE_boolean('random_transform', True, 'True if random image transformations should be applied during training')