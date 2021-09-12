import math
import sys
from os.path import dirname, abspath, join
import numpy as np
from configparser import ConfigParser
import gym

from .car_agent import CarAgent

sys.path.append('..')
config = ConfigParser()
config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))


class AirSimCarEnv(gym.Env):
    """Custom Environment that follows airsim_gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])
        self.step_reward = bool(int(config['airsim_settings']['step_reward']))
        self.speed_reward = bool(int(config['airsim_settings']['speed_reward']))
        self.distance_reward = bool(int(config['airsim_settings']['distance_reward']))
        self.nearest_way_point_reward = bool(int(config['airsim_settings']['nearest_way_point_reward']))
        self.center_of_track_reward = bool(int(config['airsim_settings']['center_of_track_reward']))
        self.car_angle_reward = bool(int(config['airsim_settings']['car_angle_reward']))
        self.track_width = int(config['airsim_settings']['track_width'])
        self.episode_step_limit = int(config['airsim_settings']['episode_step_limit'])

        # Instantiate car agent
        self.car_agent = CarAgent()

        self.previous_distance = None

    def step(self, action):
        self.number_of_steps += 1
        # move the car according to the action
        self.car_agent.move(action)
        # log info
        info = {}
        # get observation
        observation = self.car_agent.observe(False)
        # compute reward
        reward = self._compute_reward()
        # check if done
        collision_info = self.car_agent.simGetCollisionInfo()
        done = False
        if collision_info.has_collided:
            done = True
            reward = -5
        if self.number_of_steps == self.episode_step_limit:
            done = True
            reward = 5
        return observation, reward, done, info

    def reset(self):
        self.car_agent.restart()
        self.number_of_steps = 0
        observation = self.car_agent.observe(True)
        return observation  # reward, done, info can't be included

    def close(self):
        self.car_agent.reset()
        return

    def _compute_reward(self):
        max_speed = 120
        thresh_dist = 5
        reward = 1
        nearest_way_point_dist, track_center_dist, car_angle, kmh = self.car_agent.sim_get_vehicle_state()

        if self.step_reward:
            reward += self._steps_reward(self.number_of_steps, self.episode_step_limit)
        if self.speed_reward:
            reward += self._speed_reward(max_speed, kmh)
        if self.distance_reward:
            reward += self._distance_reward(20)  # TODO: get distance sensor working (re-build airsim)
        if self.nearest_way_point_reward:
            reward += self._nearest_way_point_reward(nearest_way_point_dist, thresh_dist)
        if self.center_of_track_reward:
            reward += self._center_of_track_reward(track_center_dist, self.track_width)
        if self.car_angle_reward:
            reward += self._car_angle_reward(car_angle)

        # self._print_reward(reward, max_speed, thresh_dist, nearest_way_point_dist, track_center_dist, car_angle, kmh)

        return reward

    @staticmethod
    def _steps_reward(steps, total_steps):
        return steps/total_steps

    @staticmethod
    def _speed_reward(max_speed, kmh):
        return - 0.25 ** (max_speed / (max_speed + kmh ** 2))

    @staticmethod
    def _distance_reward(distance):
        if distance < 20:
            return - (distance / 20)
        else:
            return 0

    @staticmethod
    def _nearest_way_point_reward(nearest_way_point_dist, thresh_dist):
        if nearest_way_point_dist < thresh_dist:
            return - (nearest_way_point_dist / thresh_dist)
        else:
            return -1

    @staticmethod
    def _center_of_track_reward(track_center_dist, track_width):
        return - (track_center_dist / track_width)

    @staticmethod
    def _car_angle_reward(car_angle):
        # reward + np.cos(math.radians(car_angle))
        return - np.sin(math.radians(car_angle))

    def _print_reward(self, reward, max_speed, thresh_dist, nearest_way_point_dist, track_center_dist, car_angle, kmh):

        print(
            "Speed: {} \n "
            "Nearest Way Point Dist: {} \n "
            "Track Center Dist: {} \n "
            "Car angle: {} \n "
            .format(
                kmh,
                nearest_way_point_dist,
                track_center_dist,
                car_angle))

        print(
            "Speed Reward: {} \n"
            "Nearest Way Point Reward: {} \n"
            "Track Center Reward: {} \n"
            "Car angle reward: {} \n"
            .format(
                1 - AirSimCarEnv._speed_reward(1, max_speed, kmh) if self.speed_reward else 0,
                1 - AirSimCarEnv._nearest_way_point_reward(nearest_way_point_dist, thresh_dist) if self.nearest_way_point_reward else 0,
                1 - AirSimCarEnv._center_of_track_reward(reward, track_center_dist) if self.center_of_track_reward else 0,
                1 - AirSimCarEnv._car_angle_reward(reward, car_angle) if self.car_angle_reward else 0))

        print(' \n Total Reward: {}'.format(reward))
