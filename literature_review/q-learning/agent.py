import random
import numpy as np


class QAgent:
    def __init__(self, env, discount_rate=0.95, learning_rate=0.01):
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        print("Action size:", self.action_size)
        print("State size:", self.state_size)

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])

        self.max_q = []

    def get_action(self, state):
        q_state = self.q_table[state]
        self.max_q.append(np.max(q_state))
        action_greedy = np.argmax(q_state)
        action_random = random.choice(range(self.action_size))
        return action_random if random.random() < self.eps else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.eps = self.eps * 0.98
