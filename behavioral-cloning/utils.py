import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from collections import deque

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 124, 124, 3
NUMBER_OF_ACTIONS = 11
MAX_STEERING_ANGLE = 0.35
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
OUTPUT = [0] * NUMBER_OF_ACTIONS


class ImageProcessing:

    def __init__(self):
        self.stacked_frames = None
        discrete_delta = MAX_STEERING_ANGLE/((NUMBER_OF_ACTIONS-1)/2)
        for i in range(NUMBER_OF_ACTIONS):
            OUTPUT[i] = -MAX_STEERING_ANGLE + (i * discrete_delta)


    @staticmethod
    def crop(image):
        """
        Crop the image (removing the sky at the top)
        """
        # remove the sky
        return image[54:144, :, :]

    @staticmethod
    def resize(image):
        """
        Resize the image to the input shape used by the network model
        """
        return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

    @staticmethod
    def rgb2grey(image):
        """
        Convert the image from RGB to Grey Scale
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def preprocess(self, image):
        """
        Combine all preprocess functions into one
        """
        image = self.crop(image)
        image = self.resize(image)
        image = self.rgb2grey(image)
        return image

    def stack_frames(self, frame, is_new_episode):
        """
        Stack frames into (90,256,3) - (height x width x frame stack)
        """
        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque([np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.int) for i in range(3)],
                                        maxlen=3)

            # Because we're in a new episode, copy the same frame 3x
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state

    @staticmethod
    def load_image(image_file):
        """
        Load RGB images from a file
        """
        return mpimg.imread(
            os.path.join(
                'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\training_data\\continuous_training_set_six\\'
                'converted_images_{}x{}\\'.format(IMAGE_HEIGHT, IMAGE_WIDTH),
                image_file[0]
            )
        )

    @staticmethod
    def random_flip(image, steering_angle):
        """
        Randomly flipt the image left <-> right, and adjust the steering angle.
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -int(steering_angle)
        return image, str(steering_angle)

    def augment(self, img_path, steering_angle):
        """
        Generate an augmented image and adjust steering angle.
        (The steering angle is associated with the center image)
        """
        image = self.load_image(img_path)
        image, steering_angle = self.random_flip(image, steering_angle)
        return image, steering_angle

    @staticmethod
    def convert_to_discrete_y(steering_angle):
        """
        convert from continuous to discrete
        """
        steering_angle = float(steering_angle)
        action = [0] * NUMBER_OF_ACTIONS
        action[OUTPUT.index(min(OUTPUT, key=lambda x: abs(x - steering_angle)))] = 1

        return action

    def batch_generator(self, image_paths, steering_angles, batch_size):
        """
        Generate training image give image paths and associated steering angles
        """
        images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        steers = np.empty([batch_size, NUMBER_OF_ACTIONS])
        while True:
            i = 0
            for index in np.random.permutation(image_paths.shape[0]):
                img_path = image_paths[index]
                steering_angle = steering_angles[index]

                # randomly augment data
                if np.random.rand() < 0.6:
                    image, steering_angle = self.augment(img_path, steering_angle)

                else:
                    try:
                        # load image
                        image = self.load_image(img_path)
                    except:
                        print(img_path)

                # add the image and steering angle to the batch
                images[i] = image
                steers[i] = self.convert_to_discrete_y(steering_angle)
                i += 1
                if i == batch_size:
                    break

            yield images, steers
