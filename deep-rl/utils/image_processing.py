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

    def augment(self, old_state, new_state, action, range_x=20, range_y=5):
        """
        Generate an augment image and adjust steering angle.
        (The steering angle is associated with the center image)
        """
        old_state, new_state, action = self._random_flip(old_state, new_state, action)
        old_state, new_state, action = self._random_translate(old_state, new_state, action, range_x, range_y)
        old_state, new_state = self._random_shadow(old_state, new_state)
        old_state, new_state = self._random_brightness(old_state, new_state)
        # TODO: create _random_rotate
        # old_state, new_state = self._random_rotate(old_state, new_state)

        cv2.imshow('FPV-img', new_state)
        cv2.waitKey(1)

        return old_state, new_state, action

    @staticmethod
    def _grey_scale(image):
        """
        Convert image from rgb image to grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Nvidia does this:
        # return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    @staticmethod
    def _crop(image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom)
        """
        return image[54:144, :]/255

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
            stacked_state = np.stack(self.stacked_frames, axis=0)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=0)

        return stacked_state

    def _random_flip(self, old_state, new_state, action):
        """
        Randomly flip the image left <-> right, and adjust the steering angle.
        """
        if np.random.rand() < 0.5:
            old_state = cv2.flip(old_state, 1)
            new_state = cv2.flip(new_state, 1)
            action = (self.act_dim-1) - action
        return old_state, new_state, action

    def _random_translate(self, old_state, new_state, action, range_x, range_y):
        """
        Randomly shift the image vertically and horizontally (translation).
        """
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)

        steering_angle = self.steering_values[action]
        steering_angle += trans_x * 0.002
        array = np.asarray(self.steering_values)
        action = (np.abs(array - steering_angle)).argmin()

        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = old_state.shape[:2]

        old_state = cv2.warpAffine(old_state, trans_m, (width, height))
        new_state = cv2.warpAffine(new_state, trans_m, (width, height))

        return old_state, new_state, action,

    def _random_shadow(self, old_state, new_state):
        """
        Generates and adds random shadow
        """
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = self.image_width * np.random.rand(), 0
        x2, y2 = self.image_width * np.random.rand(), self.image_height
        xm, ym = np.mgrid[0:self.image_height, 0:self.image_width]

        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line:
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = np.zeros_like(old_state[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        # hls_old = cv2.cvtColor(old_state, cv2.COLOR_RGB2HLS)
        old_state[:, :, 1][cond] = old_state[:, :, 1][cond] * s_ratio

        # hls_new = cv2.cvtColor(new_state, cv2.COLOR_RGB2HLS)
        old_state[:, :, 1][cond] = old_state[:, :, 1][cond] * s_ratio

        # old_state = cv2.cvtColor(old_state, cv2.COLOR_HLS2RGB)
        # new_state = cv2.cvtColor(hls_new, cv2.COLOR_HLS2RGB)

        return old_state, new_state

    @staticmethod
    def _random_brightness(old_state, new_state):
        """
        Randomly adjust brightness of the image.
        """
        rand = np.random.rand()

        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        ratio = 1.0 + 0.4 * (rand - 0.5)
        old_state[:, :, 2] = old_state[:, :, 2] * ratio

        ratio = 1.0 + 0.4 * (rand - 0.5)
        new_state[:, :, 2] = new_state[:, :, 2] * ratio

        return old_state, new_state

