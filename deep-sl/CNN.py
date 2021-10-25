from os.path import dirname, abspath, join
from configparser import ConfigParser
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from batch_generator import BatchGenerator
from model import Model
import json
from keras_lr_finder import LRFinder
import numpy as np


class CNN:
    def __init__(self, args, save_dir):
        self.model = None

        self.model_type = args.model_type

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), "airsim_gym", 'config.ini'))

        state_height = int(config['car_agent']['state_height'])
        state_width = int(config['car_agent']['state_width'])
        consecutive_frames = int(config['car_agent']['consecutive_frames'])
        self.act_dim = int(config['car_agent']['act_dim'])
        self.state_dim = (state_height, state_width, consecutive_frames)

        self.save_dir = save_dir
        if hasattr(args, "data_path"):
            self.data_path = args.data_path

    def create_model(self):
        model = Model(self.model_type, self.state_dim, self.act_dim)
        model.build_model()
        self.model = model

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def run_lr_finder(self, X_train, X_valid, y_train, y_valid, batch_size, epochs, samples_per_epoch):

        input_images = []
        steer_output = []

        generator = BatchGenerator(self.data_path)

        for image_name in X_train:
            image = generator.load_image(image_name)
            input_images.append(image)

        for steer in y_train:
            output = generator.convert_to_discrete_y(steer)
            steer_output.append(output)

        input_images = np.array(input_images, dtype=np.int)
        steer_output = np.array(steer_output, dtype=np.int)

        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(),
                           metrics=['accuracy'])

        # Instantiate the Learning Rate Range Test / LR Finder
        lr_finder = LRFinder(self.model)

        # Perform the Learning Rate Range Test
        # lr_finder.find(input_images, steer_output, start_lr=1e-6, end_lr=1e-2, batch_size=batch_size,
        #                epochs=epochs)

        lr_finder.find_generator(generator.batch_generator(X_train, y_train, batch_size=batch_size),
                                 start_lr=1e-6,
                                 end_lr=1e-2,
                                 epochs=epochs,
                                 steps_per_epoch=samples_per_epoch,
                                 )

        lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)

        learning_rates = lr_finder.lrs
        losses = lr_finder.losses
        json_res = []

        for i in range(len(learning_rates)):
            el = []
            el.append(learning_rates[i])
            el.append(losses[i])
            json_res.append(el)

        with open('./data_{}.json'.format(batch_size), 'w') as f:
            json.dump(str(json_res), f, indent=4)

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

        checkpoint_save_dir = join(dirname(abspath(__file__)), self.save_dir, "checkpoints",
                                   "model.{epoch:02d}-{val_loss:.2f}.h5")
        best_model_save_dir = join(dirname(abspath(__file__)), self.save_dir, "best_model", "best_model.h5")
        tensorboard_save_dir = join(dirname(abspath(__file__)), self.save_dir, "tensorboard")

        checkpoint = ModelCheckpoint(
            filepath=checkpoint_save_dir,
            verbose=0,
            save_best_only=False,
            mode='auto'
        )

        best_model = ModelCheckpoint(
            filepath=best_model_save_dir,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='auto'
        )

        callback = TensorBoard(
            log_dir=tensorboard_save_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch',
            embeddings_freq=0,
        )

        generator = BatchGenerator(self.data_path)

        # calculate the difference between expected steering angle and actual steering angle
        # square the difference
        # add up all those differences for as many data points as we have
        # divide by the number of them
        # that value is our mean squared error! this is what we want to minimize via
        # gradient descent
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

        # Fits the model on data generated batch-by-batch by a Python generator.

        # The generator is run in parallel to the model, for efficiency.
        # For instance, this allows you to do real-time data augmentation on images on CPU in
        # parallel to training your model on GPU.
        # so we reshape our data into their appropriate batches and train our model simulatenously
        history = self.model.fit_generator(
            generator.batch_generator(X_train, y_train, batch_size=batch_size),
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            max_queue_size=10,
            validation_data=generator.batch_generator(X_valid, y_valid, batch_size),
            validation_steps=100,
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0,
            callbacks=[callback, best_model]
        )

        return history

    def predict(self, state):
        return self.model.predict(state, batch_size=1)
