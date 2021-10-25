import random
import numpy as np

from collections import deque


class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size):
        """ Initialization
        """

        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state):
        """
        Save an experience to memory, optionally with its TD-Error
        """

        experience = (state, action, reward, done, new_state)

        # Check if buffer is already full
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample randomly from Buffer
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx

    def clear(self):
        """ Clear buffer / Sum Tree
        """

        self.buffer = deque()
        self.count = 0
