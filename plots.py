from app_args import FLAGS

import os
import numpy as np
import matplotlib as mpl
# For plotting without a screen
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_distribution(data, title, bins = 'auto', save_path = None, show = False):
    fig = plt.figure(figsize = (15, 6))
    plt.hist(data, bins = bins)
    plt.title(title)
    fig.text(0.9, 0.9, '{} measurements'.format(len(data)),
            verticalalignment='top', 
            horizontalalignment='center',
            color = 'black', fontsize = 12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_history(model_name, history):
    """
    Plots (and saves) the history of losses for the given model. The history object expected is the one returned
    by the keras.fit method.
    """
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
    
    plt.title("{} (EP: {}, BS: {}, LR: {}, DO: {}, BN: {})".format(
        model_name, FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout,
        '{}'.format(FLAGS.batch_norm if FLAGS.batch_norm > 0 else 'OFF')
    ))
    
    fig.text(0.5, 0, text,
                verticalalignment='top', 
                horizontalalignment='center',
                color='black', fontsize=10)
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc = (0.7, 0.5))
    fig.tight_layout()
    
    fig.savefig(os.path.join('models', model_name), bbox_inches = 'tight') 