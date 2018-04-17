import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data_loader import DataLoader
from image_processor import output_shape

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten, Convolution2D, Dropout
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

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 3, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")
flags.DEFINE_string('loss', 'mse', 'The loss function')
flags.DEFINE_float('dropout', 0.2, 'The dropout probabilty')

def build_model(input_shape, dropout_prob = FLAGS.dropout, activation = 'relu'):
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
    x = Dropout(p = dropout_prob)(x)
    
    x = Dense(100, activation = activation)(x)
    x = Dense(50, activation = activation)(x)
    x = Dense(10, activation = activation)(x)

    # The output is the steering angle
    model_out = Dense(1)(x)

    return Model(input = model_in, output = model_out)

def plot_history(model_name, history):
    
    train_log = history.history['loss']
    valid_log = history.history['val_loss']
    
    train_loss = train_log[-1]
    valid_loss = valid_log[-1]
    
    text = 'Training/Validation Loss: {:.3f} / {:.3f}'.format(train_loss, valid_loss)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    c1 = colors[0]
    c2 = colors[1]
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    ax1.set_xlabel('Epochs')    
    ax1.set_ylabel('Loss')

    x = np.arange(1, len(train_log) + 1)
    
    ax1.plot(x, train_log, label='Train Loss', color = c1)
    ax1.plot(x, valid_log, label='Validation Loss', color = c2)
    
    plt.title("{} (EP: {}, BS: {}, LR: {}, KP: {})".format(
        model_name, FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout
    ))
    
    fig.text(0.5, 0, text,
                verticalalignment='top', 
                horizontalalignment='center',
                color='black', fontsize=10)
    
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc = (0.7, 0.5))
    fig.tight_layout()
    
    fig.savefig(os.path.join('models', model_name), bbox_inches = 'tight')
    
    #plt.show()

def main(_):
    data_loader = DataLoader(TRAIN_FILE, LOG_FILE, IMG_DIR)

    images, measurements = data_loader.load_dataset()

    print('Total samples: {}'.format(images.shape[0]))

    # Split in training and validation
    X_train, X_valid, Y_train, Y_valid = train_test_split(images, measurements, 
                                                          test_size = 0.2, 
                                                          random_state = 13)

    print('Training samples: {}'.format(X_train.shape[0]))
    print('Validation samples: {}'.format(X_valid.shape[0]))

    train_generator = data_loader.generator(X_train, Y_train, FLAGS.batch_size)
    valid_generator = data_loader.generator(X_valid, Y_valid, FLAGS.batch_size)

    model = build_model(output_shape())

    model.compile(optimizer = Adam(lr = FLAGS.learning_rate), loss = FLAGS.loss)

    date_time_str = time.strftime('%Y%m%d-%H%M%S')

    callbacks = [
        TensorBoard(log_dir = os.path.join(LOGS_DIR, date_time_str), 
                    histogram_freq = 0,
                    write_graph = False,
                    write_images = False),
        EarlyStopping(monitor='val_loss', 
                    patience = 3,
                    verbose = 0, 
                    mode = 'min')
    ]
    
    print('Training on {} samples (EP: {}, BS: {}, LR: {}, DO: {})...'.format(
        X_train.shape[0], FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout
    ))
    
    history = model.fit_generator(train_generator, 
                                  nb_epoch = FLAGS.epochs,
                                  samples_per_epoch = X_train.shape[0],
                                  validation_data = valid_generator,
                                  nb_val_samples = X_valid.shape[0],
                                  callbacks = callbacks)

    model_name = 'model_{}'.format(date_time_str)

    model.save(os.path.join(MODELS_DIR, model_name + '.h5'))

    plot_history(model_name, history)

if __name__ == '__main__':
    tf.app.run()