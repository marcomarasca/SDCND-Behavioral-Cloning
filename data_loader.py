import os
import csv
import numpy as np
import cv2
import pickle
from image_processor import process_image
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataLoader():

    def __init__(self, train_file, log_file, img_folder, 
                 angle_correction = 0.2, 
                 path_separator = '\\',
                 flip_min_angle = 0.0):
        self.train_file = train_file
        self.log_file = log_file
        self.img_folder = img_folder
        self.angle_correction = angle_correction
        self.path_separator = path_separator
        self.flip_min_angle = flip_min_angle

    def load_dataset(self, regenerate = False):
        if regenerate or not os.path.isfile(self.train_file):
            print('Processing data...')
            images, measurements = self._process_data()
            self._save_pickle(images, measurements)
        else:
            print('Training file exists, loading...')
            images, measurements = self._read_pickle()
        
        return images, measurements
    
    def generator(self, images, measurements, batch_size = 64):
        
        num_samples = len(images)
        
        assert(num_samples == len(measurements))
        
        while True:
            images, measurements = shuffle(images, measurements)
            for offset in range(0, num_samples, batch_size):
                images_batch = images[offset:offset + batch_size]
                measurements_batch = measurements[offset:offset + batch_size]
                
                X_batch = np.array(list(map(self._load_image, images_batch)))
                Y_batch = measurements_batch[:,0] # Takes the steering angle only, for now
                
                yield X_batch, Y_batch

    def plot_distribution(self, data, title, save_path = None, bins = 21, show = True):
        plt.figure(figsize = (15, 6))
        plt.hist(data, bins = bins)
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

    def _load_image(self, image_file):
        img = cv2.imread(os.path.join(self.img_folder, image_file))
        img = process_image(img)
        return img
   
    def _process_data(self):
        
        images, measurements = self._load_data_log()
        images, measurements = self._flip_images(images, measurements)
        
        return images, measurements

    def _load_data_log(self):
        
        images = []
        measurements = []

        with open(self.log_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in tqdm(reader, unit = ' lines', desc = 'CSV Processing'):
                line_images, line_measurements = self._parse_line(line)

                images.extend(line_images)
                measurements.extend(line_measurements)
                
        return np.array(images), np.array(measurements)

    def _parse_line(self, line):

        images = []
        measurements = []

        center_img, left_img, right_img = [img.split(self.path_separator)[-1] for img in line[0:3]]
        steering_angle, throttle, break_force = [float(value) for value in line[3:6]]

        # Center image
        images.append(center_img)
        measurements.append((steering_angle, throttle, break_force))
        # Left image
        images.append(left_img)
        measurements.append((steering_angle + self.angle_correction, throttle, break_force))
        # Right image
        images.append(right_img)
        measurements.append((steering_angle - self.angle_correction, throttle, break_force))
        # Clips the angles to the right interval (-1, 1)
        measurements = np.clip(measurements, a_min = -1.0, a_max = 1.0)

        return images, measurements

    def _flip_images(self, images, measurements):
        
        new_images = []
        new_measurements = []
        
        for image, measurement in zip(tqdm(images, unit=' images', desc='Flipping'), measurements):
            steering_angle, throttle, break_force = measurement
            if steering_angle >= self.flip_min_angle or steering_angle <= -self.flip_min_angle:
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

