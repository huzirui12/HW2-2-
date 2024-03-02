from collections import defaultdict
import numpy as np

"""
TODO: Implement all the apply_gradients for the 3 optimizers:
    - BasicOptimizer
    - RMSProp
    - Adam
"""

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, weights, grads):
        for i in range(len(weights)):
            if not weights[i].trainable: continue
            weights[i] -= grads[i] * self.learning_rate

""" 
    def apply_gradients(self, weights, grads):
        for w, g in zip(weights, grads):
            if w.trainable:
                w -= self.learning_rate * g """


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, weights, grads):
        ## TODO: Implement RMSProp optimization
        ## HINT: Lab 2?
        # Ensure self.v is initialized here as shown in the previous response


        for i in range(len(weights)):
            if not weights[i].trainable:
                continue

            # Convert memoryview to a numpy array if necessary
            if isinstance(grads[i], memoryview):
                grads[i] = np.array(grads[i])

            # Calculate the moving average of the squared gradients
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * np.square(grads[i])

            # Update weights
            # Element-wise division by the square root of the moving average, plus epsilon
            weight_update = self.learning_rate * grads[i] / (np.sqrt(self.v[i]) + self.epsilon)

            # Assuming weights[i].data is the numpy array you want to update
            weights[i] -= weight_update


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.amsgrad = amsgrad

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.m_hat = defaultdict(lambda: 0)     # Expected value of first moment vector
        self.v_hat = defaultdict(lambda: 0)     # Expected value of second moment vector
        self.t = 0                              # Time counter

    def apply_gradients(self, weights, grads):
        ## TODO: Implement Adam optimization
        ## HINT: Lab 2?
        self.t += 1  # Increment time step
        for i in range(len(weights)):
            if not weights[i].trainable:
                continue
            if isinstance(grads[i], memoryview):
                grad_array = np.array(grads[i])
            else:
                grad_array = grads[i]

            # Update biased first moment estimate
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad_array
            # Update biased second raw moment estimate
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * np.square(grad_array)

            if self.amsgrad:
                # Update the maximum of the second moment vector
                self.v_hat[i] = np.maximum(self.v_hat[i], self.v[i])
                # Compute bias-corrected first and second moment estimates
                m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v_hat[i] / (1 - self.beta_2 ** self.t)
            else:
                # Compute bias-corrected first and second moment estimates
                m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

            # Update weights
            weight_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            weights[i] -= weight_update
