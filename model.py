import os
import time
import tensorflow as tf
import numpy as np
import image_processor as ip
import plots

from app_args import FLAGS
from data_loader import DataLoader

# Keras
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten, Convolution2D, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping

DATA_DIR   = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.p')
LOG_FILE   = os.path.join(DATA_DIR, 'driving_log.csv')
IMG_DIR    = os.path.join(DATA_DIR, 'IMG')
LOGS_DIR   = 'logs'
MODELS_DIR = 'models'

if not os.path.isdir(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if not os.path.isdir(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def fully_connected(x, output_size, activation, batch_norm = None):
    """
    Builds a fully connected layer with the given input, with batch normalization and dropout
    """
    x = Dense(output_size)(x)

    if batch_norm is not None and batch_norm > 0:
        x = BatchNormalization(momentum = batch_norm)(x)

    x = Activation(activation)(x)

    return x

def build_model(input_shape, activation = FLAGS.activation, batch_norm = FLAGS.batch_norm, dropout_prob = FLAGS.dropout):
    '''
    Defines the keras model based on the Nvidia end-to-end paper: https://arxiv.org/pdf/1604.07316v1.pdf
    '''
    # The images captured by the simulator
    model_in = Input(shape = input_shape) # 200x66@3

    # Normalize
    x = Lambda(lambda x: x/255.0 - 0.5)(model_in)

    # First three convolutions (filter size 5, stride 2x2)
    x = Convolution2D(24, 5, 5, subsample = (2, 2), activation = activation)(x) # 98x31@24
    x = Convolution2D(36, 5, 5, subsample = (2, 2), activation = activation)(x) # 47x14@36
    x = Convolution2D(48, 5, 5, subsample = (2, 2), activation = activation)(x) #  22x5@48

    # An adddional two convolutions with smaller filters (filter size 3)
    x = Convolution2D(64, 3, 3, activation = activation)(x) #20x3@64
    x = Convolution2D(64, 3, 3, activation = activation)(x) #18x1@64

    # Flatten
    x = Flatten()(x) # 1152
    
    # Fully conected layers with batch normalization and dropout
    x = fully_connected(x, 100, activation, batch_norm = batch_norm)
    x = Dropout(p = dropout_prob)(x)

    x = fully_connected(x, 50, activation, batch_norm = batch_norm)
    x = fully_connected(x, 10, activation, batch_norm = batch_norm)

    # The output is the steering angle
    model_out = Dense(1)(x)

    return Model(input = model_in, output = model_out)

def main(_):

    print('Configuration PP: {}, R: {}, C: {}, RT: {}'.format(
        FLAGS.preprocess, 
        FLAGS.regenerate, 
        FLAGS.clahe, 
        FLAGS.random_transform)
    )

    data_loader = DataLoader(TRAIN_FILE, LOG_FILE, IMG_DIR, 
                             angle_correction = FLAGS.angle_correction,
                             mirror_min_angle = FLAGS.mirror_min_angle,
                             normalize_factor = FLAGS.normalize_factor)

    images, measurements = data_loader.load_dataset(regenerate = FLAGS.regenerate)

    print('Total samples: {}'.format(images.shape[0]))

    # Split in training and validation
    X_train, X_valid, Y_train, Y_valid = data_loader.split_train_test(images, measurements)

    print('Training samples: {}'.format(X_train.shape[0]))
    print('Validation samples: {}'.format(X_valid.shape[0]))

    plots.plot_distribution(Y_train[:,0], 'Training set distribution', save_path = os.path.join('images', 'train_distribution'))
    plots.plot_distribution(Y_valid[:,0], 'Validation set distribution', save_path = os.path.join('images', 'valid_distribution'))

    train_generator = data_loader.generator(X_train, Y_train, FLAGS.batch_size, 
                                            preprocess = FLAGS.preprocess, 
                                            random_transform = FLAGS.random_transform)
    valid_generator = data_loader.generator(X_valid, Y_valid, FLAGS.batch_size, 
                                            preprocess = FLAGS.preprocess, 
                                            random_transform = False)

    model = build_model(ip.output_shape())

    model.compile(optimizer = Adam(lr = FLAGS.learning_rate), loss = FLAGS.loss)

    date_time_str = time.strftime('%Y%m%d-%H%M%S')

    callbacks = [
        # To be used with tensorboard, creates the logs for the losses in the logs dir
        TensorBoard(log_dir = os.path.join(LOGS_DIR, date_time_str), 
                    histogram_freq = 0,
                    write_graph = False,
                    write_images = False),
        # Early stopping guard
        EarlyStopping(monitor='val_loss', 
                    patience = 3,
                    verbose = 0, 
                    mode = 'min')
    ]
    
    model_name = 'model_{}'.format(date_time_str)

    print('Training {} on {} samples (EP: {}, BS: {}, LR: {}, DO: {}, BN: {}, A: {}, L: {})...'.format(
        model_name, X_train.shape[0], FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout, 
        '{}'.format(FLAGS.batch_norm if FLAGS.batch_norm > 0 else 'OFF'),
        FLAGS.activation, FLAGS.loss
    ))
    
    # Train the model
    history = model.fit_generator(train_generator, 
                                  nb_epoch = FLAGS.epochs,
                                  samples_per_epoch = X_train.shape[0],
                                  validation_data = valid_generator,
                                  nb_val_samples = X_valid.shape[0],
                                  callbacks = callbacks)

    model.save(os.path.join(MODELS_DIR, model_name + '.h5'))

    plots.plot_history(model_name, history)

if __name__ == '__main__':
    tf.app.run()
