import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from utils.path import get_export_path

from deep_sl.cnn import CNN


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--lr_range_test', type=bool, default=False, help="Run learning rate range test")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--samples_per_epoch', type=int, default=100, help="Number of  training samples per epoch")
    parser.add_argument('--lr', type=float, default=5.0e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--load_weights', type=bool, default=False, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--track', type=str, default="Track_Two")
    parser.add_argument('--model_type', type=str, default='NatureCNN', help="CNN model type to train")
    parser.add_argument('--experiment_name', type=str, default='network_architecture_study/discrete', help="CNN model type to train")
    #
    parser.add_argument(
        '--output_activation',
        type=str,
        # default='linear',
        default='softmax',
        help="Output activation function for the CNN")
    #
    parser.add_argument(
        '--loss',
        type=str,
        # default='mean_squared_error',
        default='categorical_crossentropy',
        help="loss function used to train the CNN")
    #
    parser.add_argument(
        '--pre_trained_model_path',
        type=str,
        default='',
        help="Name of the pre-trained model"
    )
    #
    parser.add_argument(
        '--data_path',
        type=str,
        default='H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\training_data\\discrete_data_set_track2',
        help="Name of the trained model"
    )

    parser.set_defaults(render=False)
    return parser.parse_args(args)


def keras_session_init():
    """
    Set up tf/keras session - use gpu device
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def load_data(args):
    """ Load training data where x is a list of image paths and y is a list of the corresponding steering angles
    """
    # read CSV file into a single data frame variable
    data_df = pd.read_csv(
        os.path.join(args.data_path, 'data.csv'),
        delimiter=';',
        names=['throttle', 'steering', 'break', 'speed', 'img']
    )

    X = data_df[['img']].values
    y = data_df['steering'].values

    # split the data into a training (80%), testing(20%), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    return X_train, X_valid, y_train, y_valid


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    keras_session_init()

    save_dir = ''
    if not args.lr_range_test:
        save_dir = get_export_path(args)

    # Load data
    data = load_data(args)

    # Parse arguments
    cnn = CNN(args, save_dir, True)
    cnn.create_model()

    # Load pre-trained model
    if args.load_weights:
        cnn.model.load_weights(os.path.join(os.getcwd(), "results", args.pre_trained_model))

    if not args.lr_range_test:
        # Train model
        cnn.train_model(*data)

    if args.lr_range_test:
        # run LR range test
        cnn.run_lr_finder(*data)


if __name__ == '__main__':
    main()
