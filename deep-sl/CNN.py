from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.layers import Lambda, Conv2D, Dropout, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import ImageProcessing, INPUT_SHAPE, NUMBER_OF_ACTIONS
from matplotlib import pyplot

process = ImageProcessing()

class CNN():
    def __init__(self):
        self.model = None

    def create_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(filters=16, kernel_size=8, strides=(4, 4), padding="valid", activation='elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding="valid", activation='elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="valid", activation='elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=4, strides=(1, 1), padding="valid", activation='elu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(256, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(NUMBER_OF_ACTIONS, activation='softmax', kernel_initializer=RandomUniform()))
        model.add(Lambda(lambda i: i * 1))
        model.summary()
        self. model = model

        return self.model

    def train_model(self, X_train, X_valid, y_train, y_valid, batch_size, epochs, samples_per_epoch):
        """
        Train the model
        """
        # Saves the model after every epoch.
        # quantity to monitor, verbosity i.e logging mode (0 or 1),
        # if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
        # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
        # made based on either the maximization or the minimization of the monitored quantity. For val_acc,
        # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
        # inferred from the name of the monitored quantity.

        # checkpoint = ModelCheckpoint(
        #     filepath="./models/4xConvLayer_1xFCLayer-84x84-image-11-discrete-actions-model.{epoch:02d}-{val_loss:.2f}.hdf5",
        #     monitor='val_loss',
        #     verbose=0,
        #     save_best_only=True,
        #     mode='auto'
        # )

        # calculate the difference between expected steering angle and actual steering angle
        # square the difference
        # add up all those differences for as many data points as we have
        # divide by the number of them
        # that value is our mean squared error! this is what we want to minimize via
        # gradient descent
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])

        # Fits the model on data generated batch-by-batch by a Python generator.

        # The generator is run in parallel to the model, for efficiency.
        # For instance, this allows you to do real-time data augmentation on images on CPU in
        # parallel to training your model on GPU.
        # so we reshape our data into their appropriate batches and train our model simulatenously
        history = self.model.fit_generator(
            process.batch_generator(X_train, y_train, batch_size=batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            max_queue_size=10,
            validation_data=process.batch_generator(X_valid, y_valid, batch_size),
            validation_steps=len(X_valid)/10,
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0,
        )

        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        # plot accuracy during training

        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(history.history['acc'], label='train')
        pyplot.plot(history.history['val_acc'], label='test')
        pyplot.legend()
        pyplot.show()

    def predict_steering(self, state):
        steering = self.model.predict(state, steps=1)
        return steering
