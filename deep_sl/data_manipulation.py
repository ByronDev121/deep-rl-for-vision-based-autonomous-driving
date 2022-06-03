import os
import pandas as pd
import cv2
from os.path import dirname, abspath, join
from configparser import ConfigParser
from utils.image_processing import ImageProcessing

config = ConfigParser()
config.read(join(dirname(dirname(abspath(__file__))), 'airsim_gym', 'config.ini'))
#
state_height = int(config['car_agent']['state_height'])
state_width = int(config['car_agent']['state_width'])
consecutive_frames = int(config['car_agent']['consecutive_frames'])
act_dim = int(config['car_agent']['act_dim'])
max_steering_angle = float(config['car_agent']['max_steering_angle'])

image_processing = ImageProcessing(state_height, state_width, consecutive_frames)


def load():
    data_df = pd.read_csv(
        os.path.join('C:\\Users\\toast\\Documents\\AirSim\\2022-02-24-11-14-25\\test\\', 'data.csv'),
        delimiter=';',
        names=['throttle', 'steering', 'break', 'speed', 'img'],
    )

    x_original = data_df[['img']].values

    for index, img_name in enumerate(x_original):
        img = cv2.imread(
            os.path.join('C:\\Users\\toast\\Documents\\AirSim\\2022-02-24-11-14-25\\test\\',
                         img_name[0])
        )
        if index == 0:
            converted_img = image_processing.preprocess(img, True)
        else:
            converted_img = image_processing.preprocess(img, False)

        # cv2.imshow("", converted_img)
        # cv2.waitKey(50)

        cv2.imwrite(
            'C:\\Users\\toast\\Documents\\AirSim\\2022-02-24-11-14-25\\test\\'
            'stacked\\{}'.format(img_name[0]),
            converted_img
        )


if __name__ == '__main__':
    load()
