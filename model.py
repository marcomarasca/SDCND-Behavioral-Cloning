import os
import time

import tensorflow as tf

from data_loader import DataLoader
from image_processor import output_shape

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten, Convolution2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.p')
LOG_FILE = os.path.join(DATA_DIR, 'driving_log.csv')
IMG_DIR = os.path.join(DATA_DIR, 'IMG')
LOGS_DIR = './logs'

# TODO Create logs dir if does not exist

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 3, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")
flags.DEFINE_string('loss', 'mse', 'The loss function')

def build_model(input_shape):
    '''
    Defines the keras model based on the Nvidia end-to-end paper: https://arxiv.org/pdf/1604.07316v1.pdf
    '''
    # The images captured by the simulator
    model_in = Input(shape = input_shape)

    # TODO Test if processing it here would speed up with GPU
    # model_in = Input(shape = (160, 320, 3))
    # x = Cropping2D(cropping = ((50, 20), (0, 0)))(model_in)
    # x = Lambda(lambda x: tf.image.resize_images(x, (200, 66), method = tf.image.ResizeMethod.AREA))(x)
    # x = Lambda(lambda x: x/255.0 - 0.5)(x)

    # Normalize
    x = Lambda(lambda x: x/255.0 - 0.5)(model_in)

    # First three convolutions (filter size 5, stride 2x2)
    x = Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu')(x)
    x = Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu')(x)
    x = Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu')(x)

    # An adddional two convolutions with smaller filters (filter size 3)
    x = Convolution2D(64, 3, 3)(x)
    x = Convolution2D(64, 3, 3)(x)

    # Flatten
    x = Flatten()(x)

    x = Dropout(p = 0.5)(x)
    x = Dense(100)(x)
    x = Dropout(p = 0.5)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)

    model_out = Dense(1)(x)

    return Model(input = model_in, output = model_out)

def main(_):
    data_loader = DataLoader(TRAIN_FILE, LOG_FILE, IMG_DIR)

    # Loads the dataset
    images, measurements = data_loader.load_dataset()

    print('Total samples: {}'.format(images.shape[0]))

    # Split in training and validation
    X_train, X_valid, Y_train, Y_valid = train_test_split(images, measurements, 
                                                        test_size = 0.2, 
                                                        random_state = 13)

    print('Training samples: {}'.format(X_train.shape[0]))
    print('Validation samples: {}'.format(X_valid.shape[0]))

    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    loss = FLAGS.loss

    train_generator = data_loader.generator(X_train, Y_train, batch_size)
    valid_generator = data_loader.generator(X_valid, Y_valid, batch_size)

    model = build_model(output_shape())

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss)

    # print(model.summary())

    date_time_str = time.strftime('%Y%m%d-%H%M%S')

    # Used for tensor board visualizations
    callbacks = [TensorBoard(log_dir = os.path.join(LOGS_DIR, date_time_str), 
                                  histogram_freq = 0,
                                  write_graph = True,
                                  write_images = False)]
    
    history = model.fit_generator(train_generator, 
                                  nb_epoch = epochs,
                                  samples_per_epoch = X_train.shape[0],
                                  validation_data = valid_generator,
                                  nb_val_samples = X_valid.shape[0],
                                  callbacks = callbacks)

    model.save('model_{}.h5'.format(date_time_str))


if __name__ == '__main__':
    tf.app.run()