import os
from data_loader import DataLoader
from keras.models import Sequential

DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.p')
LOG_FILE = os.path.join(DATA_DIR, 'driving_log.csv')
IMG_FOLDER = os.path.join(DATA_DIR, 'IMG')

def model():
    model = Sequential()

def main():
    data_loader = DataLoader(TRAIN_FILE, LOG_FILE, IMG_FOLDER)
    images, measurements = data_loader.load_dataset()

    print('Number of images: {}'.format(images.shape[0]))
    #generator = data_loader.generator(images, measurements, batch_size = 5)

if __name__ == '__main__':
    main()