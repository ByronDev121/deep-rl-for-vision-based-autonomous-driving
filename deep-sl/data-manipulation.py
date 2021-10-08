import os
import pandas as pd
import cv2
from utils import ImageProcessing, IMAGE_HEIGHT, IMAGE_WIDTH

processing = ImageProcessing()


def load():
    data_df = pd.read_csv(
        os.path.join(os.getcwd(), '..', 'training_data', 'continuous_training_set_six', 'data.csv'),
        delimiter=';',
        names=['throttle', 'steering', 'break', 'speed', 'img'],
    )

    x_original = data_df[['img']].values

    for index, img_name in enumerate(x_original):
        img = cv2.imread(
            os.path.join('C:\\Users\\toast\\Documents\\AirSim\\2020-10-12-19-13-59\\images\\',
            img_name[0])
        )
        img = processing.preprocess(img)
        if index == 0:
            converted_img = processing.stack_frames(img, True)
        else:
            converted_img = processing.stack_frames(img, False)
        cv2.imshow("", converted_img)
        cv2.waitKey(1)
        cv2.imwrite(
            'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\training_data\\continuous_training_set_six\\'
            'converted_images_{}x{}\\{}'.format(IMAGE_HEIGHT, IMAGE_WIDTH, img_name[0]),
            converted_img
        )


if __name__ == '__main__':
    load()

