import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import sys
from CNN import CNN
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_ACTIONS


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--samples_per_epoch', type=int, default=100, help="Number of  training samples per epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=3, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--load_weights', type=bool, default=True, help="Number of consecutive frames (action repeat)")
    parser.add_argument(
        '--pre_trained_model',
        type=str,
        default='4xConvLayer_1xFCLayer-{}x{}-image{}-discrete-actions-complete-training-set-model.h5',
        help="Name of the pre-trained model"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='4xConvLayer_1xFCLayer-{}x{}-image{}-discrete-actions-complete-training-set-(plus-six)-model.h5',
        help="Name of the trained model"
    )

    parser.set_defaults(render=False)
    return parser.parse_args(args)


def load_data():
    """ Load training data where x is a list of image paths and y is a list of the corresponding steering angles
    """
    # read CSV file into a single data frame variable
    data_df = pd.read_csv(
        os.path.join(os.getcwd(), '../training_data', 'continuous_training_set_six', 'data.csv'),
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

    # Load data
    data = load_data()

    # Parse arguments
    cnn = CNN()
    cnn.create_model()

    # Load pre-trained model
    args.pre_trained_model = args.pre_trained_model.format(IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_ACTIONS)
    if args.load_weights:
        cnn.model.load_weights(os.path.join(os.getcwd(), "models", args.pre_trained_model))

    # Train model
    cnn.train_model(*data, args.batch_size, args.epochs, args.samples_per_epoch)

    # Save model
    args.model_name = args.model_name.format(IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_ACTIONS)
    cnn.model.save_weights(os.path.join(os.getcwd(), "models", args.model_name))


if __name__ == '__main__':

    main()
