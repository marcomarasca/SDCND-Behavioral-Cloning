import os
import csv
import numpy as np
import cv2
import pickle
from sklearn.utils import shuffle
from tqdm import tqdm

class DataLoader():

    def __init__(self, train_file, log_file, img_folder, correction = 0.2):
        self.train_file = train_file
        self.log_file = log_file
        self.img_folder = img_folder
        self.correction = correction

    def load_dataset(self):
        if os.path.isfile(self.train_file):
            print('Training file exists, loading...')
            images, measurements = self._read_pickle()
        else:
            print('Training file absent, processing data...')
            images, measurements = self._process_data()
            self._save_pickle(images, measurements)
        
        return images, measurements
    
    def generator(self, images, measurements, batch_size = 128):
        
        num_samples = len(images)
        
        assert(num_samples == len(measurements))
        
        while 1:
            images, measurements = shuffle(images, measurements)
            for offset in range(0, num_samples, batch_size):
                images_batch = images[offset:offset + batch_size]
                measurements_batch = measurements[offset:offset + batch_size]
                
                X_batch = np.array(list(map(self._read_image, images_batch)))
                Y_batch = measurements_batch[:,0] # Takes the steering angle only for now
                
                yield X_batch, Y_batch

    def _read_image(self, image_file, color_space = cv2.COLOR_BGR2RGB):
        img = cv2.imread(os.path.join(self.img_folder, image_file))
        return cv2.cvtColor(img, color_space)

    def _process_data(self):
        
        images, measurements = self._load_data_log()
        images, measurements = self._pack_left_right(images, measurements)
        images, measurements = self._flip_images(images, measurements)
        
        return images, measurements

    def _load_data_log(self):
        
        images = []
        measurements = []

        with open(self.log_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in tqdm(reader, unit = ' lines', desc = 'CSV Processing'):
                # Win vs linux separator from the simulator
                if line[0].find("/") > -1:
                    sep = '/'
                else:
                    sep = '\\'
                center_img = line[0].split(sep)[-1]
                left_img = line[1].split(sep)[-1]
                right_img = line[2].split(sep)[-1]
                steering_angle, throttle, break_force = line[3:6]
                images.append((center_img, left_img, right_img))
                measurements.append((float(steering_angle), float(throttle), float(break_force)))
                
        return np.array(images), np.array(measurements)

    def _pack_left_right(self, images, measurements):
        
        new_images = []
        new_measurements = []
        
        for image, measurement in tqdm(zip(images, measurements), unit = ' images', desc = 'Left/Right Processing'):
            _, left_img, right_img = image
            steering_angle, throttle, break_force = measurement

            new_images.append(left_img)
            new_measurements.append((steering_angle + self.correction, throttle, break_force))

            new_images.append(right_img)
            new_measurements.append((steering_angle - self.correction, throttle, break_force))
        
        images_out = np.append(images[:,0], new_images, axis = 0)
        measurements_out = np.append(measurements, new_measurements, axis = 0)
        
        return images_out, measurements_out

    def _flip_images(self, images, measurements):
        
        new_images = []
        new_measurements = []
        
        for image, measurement in zip(tqdm(images, unit=' images', desc='Flipping'), measurements):
            steering_angle, throttle, break_force = measurement
            img = cv2.imread(os.path.join(self.img_folder, image))
            img_flipped = cv2.flip(img, 1)
            img_filpped_name = 'flipped_' + image
            cv2.imwrite(os.path.join(self.img_folder, img_filpped_name), img_flipped)
            new_images.append(img_filpped_name)
            new_measurements.append((-steering_angle, throttle, break_force))
        
        images_out = np.append(images, new_images, axis = 0)
        measurements_out = np.append(measurements, new_measurements, axis = 0)
                            
        return images_out, measurements_out

    def _read_pickle(self):
        with open(self.train_file, mode='rb') as f:
            train = pickle.load(f)
        return train['images'], train['measurements']

    def _save_pickle(self, images, measurements):
        results = {
            'images': images,
            'measurements': measurements
        }
        with open(self.train_file, 'wb') as f:   
            pickle.dump(results, f, protocol = pickle.HIGHEST_PROTOCOL)

