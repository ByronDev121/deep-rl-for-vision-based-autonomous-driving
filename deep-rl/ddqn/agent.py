import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from .model import Model


class Agent:
    """ Agent Class (Network) for ddqn
    """

    def __init__(self, model_type, state_dim, act_dim, lr, tau, dueling):
        self.model_type = model_type
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.tau = tau
        self.dueling = dueling

        # Initialize Deep Q-Network
        self.model = self.build_network()
        self.model.compile(Adam(lr), 'mse')

        # Build target Q-Network
        self.target_model = self.build_network()
        self.target_model.compile(Adam(lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())

        self.graph = tf.get_default_graph()

    @staticmethod
    def huber_loss(y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def build_network(self):
        """ Build Deep Q-Network
        """
        model = Model(self.model_type, self.state_dim, self.act_dim, self.dueling)
        model.build_model()
        return model

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        with self.graph.as_default():
            self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        with self.graph.as_default():
            return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        with self.graph.as_default():
            return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        return x.reshape(-1, *self.state_dim)

    def save(self, path):
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
