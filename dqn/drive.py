import numpy as np
from CNN import CNN
from gym_airsim.airsim_car_env import AirSimCarEnv
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from configparser import ConfigParser
from PIL import Image
from collections import deque


window_length = 4
input_shape = (84, 84)
stacked_frames = deque([np.zeros((input_shape[0], input_shape[1]), dtype=np.int) for i in range(4)], maxlen=4)


def stack_frames(frame, is_new_episode):
    if is_new_episode:
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)

    stacked_state = stacked_state.reshape(-1, 4, input_shape[0], input_shape[1])
    return stacked_state


def process_observation(observation):
    assert observation.ndim == 3  # (height, width, channel)
    img = Image.fromarray(observation)
    img = img.resize(input_shape).convert('L')  # resize and convert to grayscale
    processed_observation = np.array(img)
    assert processed_observation.shape == input_shape
    return processed_observation.astype('uint8')  # saves storage in experience memory


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

    config = ConfigParser()
    config.read('config.ini')
    num_actions = int(config['car_agent']['actions'])

    cnn = CNN()
    cnn.create_model(window_length, input_shape, num_actions)
    cnn.model.load_weights("logs/dqn_AirSimCarRL_weights_1000000.h5f")

    env = AirSimCarEnv()
    state = env.reset()
    state = process_observation(state)
    state = stack_frames(state, True)

    while True:
        steering_prediction = cnn.predict_steering(state)
        action = np.argmax(steering_prediction)
        print("Action: ", action)

        observation, reward, done, info = env.step(action)
        state = process_observation(observation)
        state = stack_frames(state, False)
        print("Reward:", reward)


if __name__ == '__main__':
    main()
