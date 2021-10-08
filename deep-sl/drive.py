import numpy as np
from CNN import CNN
from AirSim_Gym import Gym
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from utils import IMAGE_HEIGHT, IMAGE_WIDTH


def main():

    gym = Gym()
    gym.reset()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

    cnn = CNN()
    cnn.create_model()
    cnn.model.load_weights(
        "models/4xConvLayer_1xFCLayer-124x124-image11-discrete-actions-complete-training-set-(plus-six)-model.h5".format(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )
    )

    i = 0
    while True:
        if i == 0:
            state = gym.get_image(True)
            i = i + 1
        else:
            state = gym.get_image(False)

        if state is not None:
            steering_prediction = cnn.predict_steering(np.array(state).reshape(-1, *state.shape))
            action = np.argmax(steering_prediction)
            print(action)

            gym.act(action)


if __name__ == '__main__':
    main()
