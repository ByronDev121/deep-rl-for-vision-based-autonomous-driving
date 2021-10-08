import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Permute, Dense, Flatten, Lambda, Activation


class Model (Sequential):

    def __init__(self, model_type, state_dim, act_dim, dueling):
        super()
        super().__init__()
        self.model_type = model_type
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.dueling = dueling

    def build_model(self):
        if self.model_type == "NatureCNN":
            self.build_nature_cnn()
        elif self.model_type == "NvidiaCNN":
            self.build_nvidia_cnn()
        elif self.model_type == "CustomCNN":
            self.build_custom_cnn()
        else:
            self.build_nature_cnn()
            print(EnvironmentError('Model no recognized - building Nature CNN'))

        if self.dueling:
            # Have the network estimate the Advantage function as an intermediate layer
            self.add(Dense(units=self.act_dim + 1))
            self.add(Activation('linear'))
            self.add(Lambda(
                lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                output_shape=(self.act_dim,)
            ))

        else:
            self.add(Dense(units=self.act_dim))
            self.add(Activation('linear'))

        print(self.summary())

    # The same model that was described by Mnih et al. (2015).
    def build_nature_cnn(self):
        self.add(Permute(dims=(2, 3, 1), input_shape=self.state_dim))
        self.add(Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu'))
        self.add(Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu'))
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
        self.add(Flatten())
        self.add(Dense(units=512, activation='relu'))

    # The same model that was described by Bojarski, Mariusz, et al. (2016)
    def build_nvidia_cnn(self):
        self.add(Permute(dims=(2, 3, 1), input_shape=self.state_dim))
        self.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        self.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        self.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        self.add(Conv2D(64, 3, 3, activation='elu'))
        self.add(Conv2D(64, 3, 3, activation='elu'))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(100, activation='elu'))
        self.add(Dense(50, activation='elu'))

    def build_custom_cnn(self):
        self.add(Permute(dims=(2, 3, 1), input_shape=self.state_dim))
        self.add(Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding="valid", activation='elu'))
        self.add(Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding="valid", activation='elu'))
        self.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="valid", activation='elu'))
        self.add(Flatten())
        self.add(Dense(units=512, activation='elu'))
        self.add(Dense(units=100, activation='elu'))
        self.add(Dense(units=50, activation='elu'))
