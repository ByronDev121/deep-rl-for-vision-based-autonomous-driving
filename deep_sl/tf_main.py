#CNN Taining Script For Autonomous Car:

#This CNN has two "convpool" layers and one fully connected hidden layer.
#Implimentation using Tensorflow.


from __future__ import print_function, division
from builtins import range

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import math

HEIGHT = 50
WIDTH = 50


def y2indicator(y):
    N = len(y[:, ])
    Y_ind = np.zeros((N, 1), dtype=np.float32)
    Y_ind[:, 0] = y[:, 1]
    maximum = max(y[:, 1])
    minimum = min(y[:, 1])
    Y_ind = Y_ind / maximum
    print(maximum)
    return Y_ind


def error_rate(p, t):
    return np.mean(abs(p - t))


def error_percent(err, t):
    Max = max(t)
    Min = min(t)
    Denominator = (Max - Min)
    ERR_Percent = (err / Denominator)
    return ERR_Percent


def get_data():
    if not os.path.exists('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\training_data\\Train_37000.mat'):
        exit()
    # RoboCar Training Data X = 42000 - 50x50x3 = (50,50,3,42000) images
    train = loadmat('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\training_data\\Train_37000.mat')
    # Testing on the same data...
    test = loadmat('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\training_data\\Test_5000Split.mat')

    # Need to scale! don't leave as 0..255
    # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calculation
    Xtrain = rearrange(train['X'])
    Ytrain = np.asanyarray(train['y'], dtype=np.float32)
    Ytrain_ind = y2indicator(Ytrain)
    del train
    Xtest = rearrange(test['X'])
    Ytest = np.asanyarray(test['y'], dtype=np.float32)
    del test
    Ytest_ind = y2indicator(Ytest)

    # limit samples since input will always have to be same size
    # you could also just do N = N / batch_sz * batch_sz
    ###### This is done to save RAM --->> Use as Test
    Xtrain = Xtrain[:37000, ]
    Ytrain = Ytrain[:37000]
    Ytrain_ind = Ytrain_ind[:37000, ]
    Xtest = Xtest[:5000, ]
    Ytest = Ytest[:5000]
    Ytest_ind = Ytest_ind[:5000, ]
    # print "Xtest.shape:", Xtest.shape
    # print "Ytest.shape:", Ytest.shape

    return Xtrain, Xtest, Ytrain_ind, Ytest_ind

def y2indicator_2(y):
    N = len(y)
    Y_ind = np.zeros((N, 1), dtype=np.float32)
    Y_ind[:, 0] = y[:, 0]
    maximum = max(y[:, 0])
    minimum = min(y[:, 0])
    Y_ind = Y_ind / maximum
    print(maximum)
    return Y_ind


def load_image(image_file):
    """
    Load RGB images from a file
    """
    image = cv2.imread(
        os.path.join(
            'C:\\Users\\toast\\Documents\\AirSim\\2022-02-24-11-14-25\\images',
            image_file[0]
        ))

    if image is None:
        print('None')
        print(image_file[0])

    return cv2.resize(image, (WIDTH, HEIGHT), cv2.INTER_AREA)


def load_data():
    """ Load training data where x is a list of image paths and y is a list of the corresponding steering angles
    """
    # read CSV file into a single data frame variable
    data_df = pd.read_csv(
        os.path.join('C:\\Users\\toast\\Documents\\AirSim\\2022-02-24-11-14-25', 'data.csv'),
        delimiter=';',
        names=['throttle', 'steering', 'break', 'speed', 'img']
    )

    X = data_df[['img']].values
    y = data_df['steering'].values

    # split the data into a training (80%), testing(20%), and validation set
    X_train_strings, X_valid_strings, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    Xtrain = np.empty([len(X_train_strings), HEIGHT, WIDTH, 3])
    Xtest = np.empty([len(X_valid_strings), HEIGHT, WIDTH, 3])

    for idx, val in enumerate(X_train_strings):
        img = load_image(val)
        Xtrain[idx, :] = img / 255
    for idx, val in enumerate(X_valid_strings):
        img = load_image(val)
        Xtest[idx, :] = img / 255

    print('X_train shape', Xtrain.shape)
    print('X_valid shape', Xtest.shape)

    Ytrain = Ytrain.reshape((len(Ytrain), 1))
    Ytest = Ytest.reshape((len(Ytest), 1))

    # Xtrain = rearrange(Xtrain)
    Ytrain = np.asanyarray(Ytrain, dtype=np.float32)
    Ytrain_ind = y2indicator_2(Ytrain)

    # Xtest = rearrange(Xtest)
    Ytest = np.asanyarray(Ytest, dtype=np.float32)
    Ytest_ind = y2indicator_2(Ytest)

    return Xtrain, Xtest, Ytrain_ind, Ytest_ind


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


def rearrange(X):
    return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)


def main():
    Xtrain, Xtest, Ytrain_ind, Ytest_ind = load_data()

    # gradient descent params
    max_iter = 1000
    print_period = 10
    N = Xtrain.shape[0]
    batch_sz = 256
    n_batches = N // batch_sz

    # initial weights
    M = 500
    K = 1
    poolsz = (2, 2)

    W1_shape = (5, 5, 3, 20) # (filter_width, filter_height, num_color_channels, num_feature_maps)
    W1_init = init_filter(W1_shape, poolsz)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32) # one bias per output feature map

    W2_shape = (5, 5, 20, 50) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # vanilla ANN weights
    W3_init = np.random.randn(W2_shape[-1]*13*13, M) / np.sqrt(W2_shape[-1]*13*13 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)

    with tf.device('/gpu:0'):

        # define variables and expressions
        # using None as the first shape element takes up too much RAM unfortunately
        X = tf.placeholder(tf.float32, shape=(batch_sz, HEIGHT, WIDTH, 3), name='X')
        T = tf.placeholder(tf.float32, shape=(batch_sz, K), name='T')
        W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
        b1 = tf.Variable(b1_init.astype(np.float32), name='b1')
        W2 = tf.Variable(W2_init.astype(np.float32), name='W2')
        b2 = tf.Variable(b2_init.astype(np.float32), name='b2')
        W3 = tf.Variable(W3_init.astype(np.float32), name='W3')
        b3 = tf.Variable(b3_init.astype(np.float32), name='b3')
        W4 = tf.Variable(W4_init.astype(np.float32), name='W4')
        b4 = tf.Variable(b4_init.astype(np.float32), name='b4')

        Z1 = convpool(X, W1, b1)
        Z2 = convpool(Z1, W2, b2)
        Z2_shape = Z2.get_shape().as_list()
        Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
        Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3)

        #Output Layer:
        Yish = tf.nn.tanh(tf.matmul(Z3, W4) + b4)

        MSE_Cost = tf.losses.mean_squared_error(Yish, T)

        Train_Optomizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(MSE_Cost)

        # we'll use this to calculate the error rate
        predict_op = Yish

        t0 = datetime.now()
        Cost_plot = []
        Error_Plot = []
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as session:
            session.run(init)

            for i in range(max_iter):

                for j in range(int(n_batches)): #This could also be math.round(n_batches)
                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                    Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                    if len(Xbatch[:,]) == batch_sz:
                        session.run(Train_Optomizer, feed_dict={X: Xbatch, T: Ybatch})
                        if j % print_period == 0:
                            # due to RAM limitations we need to have a fixed size input
                            # so as a result, we have this ugly total cost and prediction computation
                            test_cost = 0
                            prediction = np.zeros((math.floor(len(Xtest) // batch_sz) * batch_sz, K), dtype=np.float32)
                            for k in range(math.floor(len(Xtest) // batch_sz)):

                                start_index = k*batch_sz
                                end_index = (k+1) * batch_sz

                                Xtestbatch = Xtest[start_index:end_index, ]
                                Ytestbatch = Ytest_ind[start_index:end_index, ]
                                test_cost += session.run(MSE_Cost, feed_dict={X: Xtestbatch, T: Ytestbatch})
                                prediction[start_index:end_index] = session.run(predict_op, feed_dict={X: Xtestbatch})

                            Error = error_rate(prediction, Ytest_ind[0:math.floor(len(Xtest) // batch_sz) * batch_sz])
                            Error_percent = error_percent(Error, Ytest_ind[0:math.floor(len(Xtest) // batch_sz) * batch_sz])
                            print("Cost / Error at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, Error_percent))
                            print("Prediction:", prediction)
                            print("Actual", Ytest_ind)
                            Cost_plot.append(test_cost)
                            Error_Plot.append(Error_percent)

                            if i == 300:
                                saver.save(session, 'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-300\\my_model', global_step=i)
                                print("Elapsed time:", (datetime.now() - t0))
                                plt.plot(Cost_plot)
                                plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-300\\Cost_plot.png', format='png')

                                plt.plot(Error_Plot)
                                plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-300\\Error_plot.png', format='png')

                                np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-300\\prediction.npy', prediction)
                                np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-300\\Ytest_ind.npy', Ytest_ind)

                            if i == 500:
                                saver.save(session,'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-500\\my_model', global_step=i)
                                print("Elapsed time:", (datetime.now() - t0))
                                plt.plot(Cost_plot)
                                plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-500\\Cost_plot.png',format='png')

                                plt.plot(Error_Plot)
                                plt.savefig( 'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-500\\Error_plot.png',format='png')

                                np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-500\\prediction.npy',prediction)
                                np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-500\\Ytest_ind.npy',Ytest_ind)

                            if i == 600:
                                saver.save(session, 'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-600\\my_model', global_step=i)
                                print("Elapsed time:", (datetime.now() - t0))
                                plt.plot(Cost_plot)
                                plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-600\\Cost_plot.png', format='png')

                                plt.plot(Error_Plot)
                                plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-600\\Error_plot.png', format='png')

                                np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-600\\prediction.npy', prediction)
                                np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-600\\Ytest_ind.npy', Ytest_ind)
                            if i == 999:
                                saver.save(session, 'H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-999\\my_model', global_step=i)

        print("Elapsed time:", (datetime.now() - t0))
        plt.plot(Cost_plot)
        plt.show()
        plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-999\\Cost_plot.png', format='png')

        plt.plot(Error_Plot)
        plt.show()
        plt.savefig('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-999\\Cost_plot.png', format='png')

        np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-999\\prediction.npy', prediction)
        np.save('H:\\Masters\\AirSim\\AirSim_Source\\PythonClient\\car\\Masters\\deep_sl\\results\\old_CNN\\CNN_Model_i-999\\Ytest_ind.npy', Ytest_ind)

if __name__ == '__main__':
    main()