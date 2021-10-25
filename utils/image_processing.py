import cv2
import numpy as np
from collections import deque


class ImageProcessing:
    """
    Convert image from rgb image to grayscale image.

    @param: image_width - Width of image in pixels
    @param: image_height - Height of image on pixels
    @param: image_channels  - Number of channels to stack frames
    """

    def __init__(self, image_height, image_width, image_channels, act_dim, max_steering_angle):
        self.stacked_frames = None
        self.image_width = image_width
        self.image_height = image_height
        self.channels = image_channels
        self.act_dim = act_dim
        self.steering_values = np.arange(
            -max_steering_angle,
            max_steering_angle,
            2 * max_steering_angle / (act_dim - 1)
        ).tolist()
        self.steering_values.append(max_steering_angle)
        self.steering_values = [round(num, 3) for num in self.steering_values]

    def preprocess(self, image, is_new_episode):
        """
        Combine all preprocess functions into one
        """
        image = self._grey_scale(image)
        image = self._crop(image)
        image = self._resize(image)
        image = self._stack_frames(image, is_new_episode)
        # cv2.imshow('FPV-img', image)
        # cv2.waitKey(1)
        return image

    @staticmethod
    def _grey_scale(image):
        """
        Convert image from rgb image to grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _crop(image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom)
        """
        return image[54:144, :] / 255

    def _resize(self, image):
        """
        Resize the image to the input shape used by the network model
        """
        return cv2.resize(image, (self.image_width, self.image_height), cv2.INTER_AREA)

    def _stack_frames(self, frame, is_new_episode):
        """
        Stack frames to give network a temporal sense of the data - if initial state (initial state is stacked)
        """
        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque(
                [np.zeros((self.image_width, self.image_height), dtype=np.int) for i in range(3)],
                maxlen=self.channels
            )

            # Because we're in a new episode, copy the same frame
            for x in range(self.channels):
                self.stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state
