import numpy as np
import cv2
import random
import math


class Augment:
    def __init__(
            self,
            state_height,
            state_width,
            random_flip,
            random_translate,
            random_rotate,
            random_depth,
            random_brightness):
        self.state_height = state_height
        self.state_width = state_width
        self.do_random_flip = random_flip
        self.do_random_translate = random_translate
        self.do_random_rotate = random_rotate
        self.do_random_depth = random_depth
        self.do_random_brightness = random_brightness

    @staticmethod
    def random_flip(image, steering_angle):
        """
        Randomly flipt the image left <-> right, and adjust the steering angle.
        """
        image = cv2.flip(image, 1)
        steering_angle = -float(steering_angle)
        return image, steering_angle

    @staticmethod
    def random_rotate(image):
        angle = random.randint(-2, 2)
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    @staticmethod
    def random_translate(image, steering_angle):
    # def random_translate(image, steering_angle, range_x=25, range_y=10):
        """
        Randomly shift the image virtially and horizontally (translation).
        """
        range_x = math.floor(len(image[0]) * 0.12)
        range_y = math.floor(len(image[1]) * 0.06)
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    @staticmethod
    def random_depth(image, state_width, state_height):
        ht, wd, cc = image.shape

        if np.random.rand() > 0.5:
            # create new image of desired size and color (blue) for padding
            ww = wd + math.floor(12 * np.random.rand())
            hh = ht + math.floor(8 * np.random.rand())

            color = (0, 0, 0)
            result = np.full((hh, ww, cc), color, dtype=np.uint8)

            # compute center offset
            xx = (ww - wd) // 2
            yy = (hh - ht) // 2

            # copy img image into center of result image
            result[yy:yy + ht, xx:xx + wd] = image
        else:

            ww = math.floor(wd - math.floor(12 * np.random.rand()))
            hh = math.floor(ht - math.floor(8 * np.random.rand()))
            x = math.floor(((wd - ww) / 2) + (((wd - ww) / 2) * np.random.rand()))
            y = math.floor(((ht - hh) / 2) + (((ht - hh) / 2) * np.random.rand()))
            result = image[y:hh, x:ww, :]

        return cv2.resize(result, (state_width, state_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def random_brightness(input_img):
        """
        Randomly adjust brightness of the image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        value = math.floor(64 * np.random.rand())
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def augment(self, image, steering_angle):
        """
        Generate an augmented image and adjust steering angle.
        (The steering angle is associated with the center image)
        """

        # cv2.imshow("image_before", image)
        # cv2.waitKey(10)

        if self.do_random_flip and np.random.rand() < 0.6:
            image, steering_angle = self.random_flip(image, steering_angle)
        if self.do_random_translate and np.random.rand() < 0.6:
            image, steering_angle = self.random_translate(image, steering_angle)
        if self.do_random_rotate and np.random.rand() < 0.6:
            image = self.random_rotate(image)
        if self.do_random_depth and np.random.rand() < 0.6:
            image = self.random_depth(image, self.state_width, self.state_height)
        if self.do_random_brightness and np.random.rand() < 0.6:
            image = self.random_brightness(image)

        # cv2.imshow("image_after", image)
        # cv2.waitKey(10)

        return image, steering_angle
