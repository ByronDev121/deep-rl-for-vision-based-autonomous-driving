from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Concatenate


class CNN:
    def __init__(self):
        self.model = None

    def create_model(self, window_length, input_shape, num_actions):
        # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
        input_shape = (window_length,) + input_shape
        self.model = Sequential()
        self.model.add(Permute((2, 3, 1), input_shape=input_shape))
        self.model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(num_actions))
        self.model.add(Activation('linear'))
        print(self.model.summary())

        return self.model

    def predict_steering(self, state):
        steering = self.model.predict(state, steps=1)
        return steering
