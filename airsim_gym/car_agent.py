import math
import numpy as np
from os.path import dirname, abspath, join
from numpy.linalg import norm
from configparser import ConfigParser
from gym import spaces

from Masters.utils.image_processing import ImageProcessing

import airsim
from airsim import CarClient, CarControls, ImageRequest, ImageType, Pose, Vector3r


class CarAgent(CarClient):

    def __init__(self):
        # connect to the AirSim simulator
        super().__init__()
        super().confirmConnection()
        super().enableApiControl(True)

        #

        config = ConfigParser()
        config.read(join(dirname(abspath(__file__)), 'config.ini'))

        way_point_regex = config['airsim_settings']['waypoint_regex']
        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])

        self._fetch_way_points(way_point_regex)
        self.airsim_image_size = self.image_height * self.image_width * self.image_channels

        #

        state_height = int(config['car_agent']['state_height'])
        state_width = int(config['car_agent']['state_width'])
        act_dim = spaces.Discrete(int(config['car_agent']['act_dim']))
        consecutive_frames = int(config['car_agent']['consecutive_frames'])
        max_steering_angle = float(config['car_agent']['max_steering_angle'])
        steering_granularity = int(config['car_agent']['steering_granularity'])
        self.action_mode = int(config['car_agent']['action_mode'])
        self.fixed_throttle = float(config['car_agent']['fixed_throttle'])
        self.random_spawn = float(config['car_agent']['random_spawn'])

        self.steering_values = self._set_steering_values(max_steering_angle, steering_granularity)
        self.image_processing = ImageProcessing(state_height, state_width, consecutive_frames, act_dim.n,
                                                max_steering_angle)

        #

        self.previous_position = np.array([0, 0])
        self.spawn_position = 0

    @staticmethod
    def _set_steering_values(max_steering_angle, steering_granularity):
        steering_values = np.arange(
            -max_steering_angle,
            max_steering_angle,
            2 * max_steering_angle / (steering_granularity - 1)
        ).tolist()
        steering_values.append(max_steering_angle)
        steering_values = [round(num, 3) for num in steering_values]
        return steering_values

    def restart(self):
        next_position = self._get_spawn_position()
        super().reset()
        super().enableApiControl(True)
        if self.random_spawn:
            super().simSetVehiclePose(next_position, True)

    def _get_spawn_position(self):
        if self.spawn_position == 0:
            self.spawn_position = 1
            return Pose(Vector3r(0.0, 0.0, 0.0), airsim.to_quaternion(0.0, 0.0, -0.1))
        elif self.spawn_position == 1:
            self.spawn_position = 2
            return Pose(Vector3r(504.6, 4.7, 0.0), airsim.to_quaternion(0.0, 0.0, 0.1))
        elif self.spawn_position == 2:
            self.spawn_position = 3
            return Pose(Vector3r(499.6, 260.4, 0.0), airsim.to_quaternion(0.0, 0.0, 3.57792))
        elif self.spawn_position == 3:
            self.spawn_position = 0
            return Pose(Vector3r(53.3, 231.7, 0.0), airsim.to_quaternion(0.0, 0.0, 2.87979))

    def observe(self, is_new):
        size = 0
        # Sometimes simGetImages() return an unexpected response.
        # If so, try it again.
        while size != self.airsim_image_size:
            response = super().simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
            img1d_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            size = img1d_rgb.size

        img3d_rgb = img1d_rgb.reshape(self.image_height, self.image_width, self.image_channels)
        processed_image = self.image_processing.preprocess(img3d_rgb, is_new)

        return processed_image

    def move(self, action):
        car_controls = self._interpret_action(action)
        super().setCarControls(car_controls)

    @staticmethod
    def _get_angle(point1, point2):
        p1 = point1 if point1[0] > point2[0] else point2
        p2 = point1 if point1[0] < point2[0] else point2
        dX = p2[0] - p1[0]
        dY = p2[1] - p1[1]
        rads = math.atan2(-dY, dX)  # wrong for finding angle/declination?
        return math.degrees(rads)

    def sim_get_vehicle_state(self):
        car_state = super().getCarState()

        # distance_sensor_data = self.getDistanceSensorData(lidar_name="", vehicle_name="")
        # distance_sensor_data = super().getDistanceSensorData(lidar_name="", vehicle_name="")

        speed = car_state.speed
        kmh = int(3.6 * speed)

        pos = super().simGetVehiclePose().position
        car_point = np.array([pos.x_val, pos.y_val])

        way_point_one, way_point_two, distance_to_nearest_way_point = self._get_two_closest_way_points(car_point)

        # Perpendicular  distance to the line connecting 2 closest way points,
        # this distance is approximate to distance to center of track
        distance_p1_to_p2p3 = lambda p1, p2, p3: abs(np.cross(p2 - p3, p3 - p1)) / norm(p2 - p3)
        distance_to_track_center = distance_p1_to_p2p3(car_point, way_point_one, way_point_two)

        angle1 = self._get_angle(way_point_one, way_point_two)
        angle2 = self._get_angle(car_point, self.previous_position)

        ang_diff = 0
        if abs(angle2) > abs(angle1) and speed > 1:
            ang_diff = abs(angle2) - abs(angle1)
        elif speed > 1:
            ang_diff = abs(angle1) - abs(angle2)

        # print("Angle difference: ", ang_diff)
        self.previous_position = car_point

        return distance_to_nearest_way_point, distance_to_track_center, ang_diff, kmh

    def _fetch_way_points(self, waypoint_regex):
        wp_names = super().simListSceneObjects(waypoint_regex)
        wp_names.sort()
        print(wp_names)
        vec2r_to_numpy_array = lambda vec: np.array([vec.x_val, vec.y_val])

        self.waypoints = []
        for wp in wp_names:
            pose = super().simGetObjectPose(wp)
            self.waypoints.append(vec2r_to_numpy_array(pose.position))

        return

    def _interpret_action(self, action):
        car_controls = CarControls()

        if self.action_mode == 0:  # discrete steering only, throttle is fixed
            car_controls.throttle = self.fixed_throttle
            car_controls.steering = self.steering_values[action]
        elif self.action_mode == 1:  # average value continuous steering only, throttle is fixed
            # filter action
            actual_action = self.steering_values[action]
            self.kf.update(actual_action)
            filtered_action = self.kf.predict()
            # print('Actual action: {} \nFiltered action: {}'.format(actual_action, filtered_action))
            car_controls.throttle = self.fixed_throttle
            car_controls.steering = filtered_action
        elif self.action_mode == 2:  # continuous steering only, throttle is fixed
            car_controls.throttle = self.fixed_throttle
            car_controls.steering = float(action)
        else:
            return NotImplemented

        return car_controls

    def _get_two_closest_way_points(self, car_point):

        min_dist = 9999999
        second_min_dist = 9999999
        min_i = 0
        second_min_i = 0
        for i in range(len(self.waypoints) - 1):
            dist = math.sqrt(
                pow((car_point[0] - self.waypoints[i][0]), 2) +
                pow((car_point[1] - self.waypoints[i][1]), 2)
            )
            if dist < min_dist:
                second_min_dist = min_dist
                second_min_i = min_i
                min_dist = dist
                min_i = i
            elif dist < second_min_dist:
                second_min_dist = dist
                second_min_i = i

        return self.waypoints[min_i], self.waypoints[second_min_i], min_dist
