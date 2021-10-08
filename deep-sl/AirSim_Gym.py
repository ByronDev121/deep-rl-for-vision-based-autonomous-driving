import airsim
import cv2
import numpy as np
from utils import ImageProcessing, OUTPUT

process = ImageProcessing()

class Gym:

    def __init__(self, show_cam=True):
        # connect to the AirSim simulator
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.SHOW_CAM = show_cam

    def get_image(self, is_new_episode):
        try:
            # RGB image
            image = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
            image1d = np.fromstring(image.image_data_uint8, dtype=np.uint8)
            image_rgb = image1d.reshape(image.height, image.width, 3)
            if image_rgb is not None:
                frame = process.preprocess(image_rgb)
                stacked_frames = process.stack_frames(frame, is_new_episode)

                if self.SHOW_CAM:
                    cv2.imshow('FPV-img', stacked_frames)
                    cv2.waitKey(1)

                return stacked_frames

        except Exception as e:
            print('could not get img')

    def act(self, action):

        collision_info = self.client.simGetCollisionInfo()

        if collision_info.has_collided:
            self.reset()

        self.car_controls.throttle = 0.65

        action = OUTPUT[action]
        self.car_controls.steering = action
        self.client.setCarControls(self.car_controls)

    def reset(self):
        self.client.reset()