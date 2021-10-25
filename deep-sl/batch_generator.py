import os
import numpy as np
import cv2
from os.path import dirname, abspath, join
from configparser import ConfigParser

from augment import Augment


class BatchGenerator:
    def __init__(self, data_path):

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'airsim_gym', 'config.ini'))
        #
        self.state_height = int(config['car_agent']['state_height'])
        self.state_width = int(config['car_agent']['state_width'])
        self.consecutive_frames = int(config['car_agent']['consecutive_frames'])
        self.act_dim = int(config['car_agent']['act_dim'])
        max_steering_angle = float(config['car_agent']['max_steering_angle'])

        random_flip = bool(int(config['car_agent']['random_flip']))
        random_translate = bool(int(config['car_agent']['random_translate']))
        random_rotate = bool(int(config['car_agent']['random_rotate']))
        random_depth = bool(int(config['car_agent']['random_depth']))
        random_brightness = bool(int(config['car_agent']['random_brightness']))
        self.do_augmentation = bool(int(config['car_agent']['augment']))

        self.stacked_frames = None
        self.augment = Augment(
            self.state_height,
            self.state_width,
            random_flip,
            random_translate,
            random_rotate,
            random_depth,
            random_brightness
        )
        self.steering_values = [0] * self.act_dim
        discrete_delta = max_steering_angle / ((self.act_dim - 1) / 2)
        for i in range(self.act_dim):
            self.steering_values[i] = -max_steering_angle + (i * discrete_delta)

        self.data_path = data_path

    def load_image(self, image_file):
        """
        Load RGB images from a file
        """
        image = cv2.imread(
            os.path.join(
                '{}\\images'.format(self.data_path),
                image_file[0]
            )
        )

        return cv2.resize(image, (self.state_width, self.state_height), cv2.INTER_AREA)

    def convert_to_discrete_y(self, steering_angle):
        """
        convert from continuous to discrete
        """
        steering_angle = float(steering_angle)
        action = [0] * self.act_dim
        action[self.steering_values.index(min(self.steering_values, key=lambda x: abs(x - steering_angle)))] = 1

        return action

    def batch_generator(self, image_paths, steering_angles, batch_size):
        """
        Generate training image give image paths and associated steering angles
        """
        images = np.empty([batch_size, self.state_height, self.state_width, self.consecutive_frames])
        steers = np.empty([batch_size, self.act_dim])
        while True:
            i = 0
            for index in np.random.permutation(image_paths.shape[0]):
                img_path = image_paths[index]
                image = self.load_image(img_path)
                steering_angle = steering_angles[index]

                # randomly augment data
                if self.do_augmentation and np.random.rand() < 0.7:
                    image, steering_angle = self.augment.augment(image, steering_angle)

                # cv2.imshow("image_after", image)
                # cv2.waitKey(1)

                # add the image and steering angle to the batch
                images[i] = image / 256
                steers[i] = self.convert_to_discrete_y(steering_angle)
                i += 1
                if i == batch_size:
                    break

            yield images, steers
