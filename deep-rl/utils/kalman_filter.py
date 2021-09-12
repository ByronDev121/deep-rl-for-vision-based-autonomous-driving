from collections import deque


class KalmanFilter1D_old:
    def __init__(self, estimate_variance, measurement_variance):
        self.current_estimate = 0
        self.previous_estimate = 0

        self.kalman_gain = 1
        self.estimate_variance = estimate_variance
        self.measurement_variance = measurement_variance

    def update(self, measurement):
        # Update previous estimate
        self.previous_estimate = self.current_estimate

        # Update the kalman gain
        self.kalman_gain = self.estimate_variance / (self.estimate_variance + self.measurement_variance)

        # Estimate the current state
        self.current_estimate = self.previous_estimate + self.kalman_gain * (measurement - self.previous_estimate)

        # Update current estimate uncertainty
        self.estimate_variance = (1 - self.kalman_gain) * self.estimate_variance

    def predict(self):
        return self.current_estimate


class KalmanFilter1D_old_1:
    def __init__(self, estimate_variance, measurement_variance):
        self.current_estimate = 0
        self.previous_estimate = 0

        self.last_three_estimates = deque()

        self.mean = 0
        self.variance = estimate_variance
        self.measurement_variance = measurement_variance
        self.process_variance = measurement_variance

    def multiply(self, measurement):
        if self.variance == 0.0:
            self.variance = 1.e-80
        if self.measurement_variance == 0:
            self.measurement_variance = 1e-80

        self.mean = (self.variance * measurement + self.measurement_variance * self.mean) / \
               (self.variance + self.measurement_variance)

        self.variance = 1 / (1 / self.variance + 1 / self.measurement_variance)

    def update(self, measurement):
        self.multiply(measurement)

    def predict(self):
        self.mean = self.mean + self.variance
        self.variance = self.variance + self.process_variance
        return self.mean


class KalmanFilter1D:
    def __init__(self, estimate_variance, measurement_variance):
        self.last_three_estimates = deque()
        self.mean = 0

    def update(self, measurement):
        if len(self.last_three_estimates) == 3:
            self.last_three_estimates.popleft()
        self.last_three_estimates.append(measurement)

    def predict(self):
        return sum(self.last_three_estimates) / len(self.last_three_estimates)