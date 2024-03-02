import numpy as np

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    """
    TODO:
        - call function
        - input_gradients
    Identical to HW1!
    """

    def call(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        self.y_pred = y_pred
        self.y_true = y_true
        return Tensor(np.mean((y_pred - y_true) ** 2))

    def get_input_gradients(self) -> list[Tensor]:
        grady_pred=Tensor(2*(self.y_pred - self.y_true)/self.y_pred.size)
        grady_true=Tensor(np.zeros_like(grady_pred))
        return [grady_pred,grady_true]


def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1 - eps)


class CategoricalCrossentropy(Loss):
    """
    TODO: Implement CategoricalCrossentropy class
        - call function
        - input_gradients

        Hint: Use clip_0_1 to stabilize calculations
    """

    def call(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred_clipped = clip_0_1(y_pred)
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
        return np.mean(loss)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        # Clip predictions to avoid division by 0
        y_pred = self.inputs[0]
        y_true = self.inputs[1]
        y_pred_clipped = clip_0_1(y_pred)
        # Compute the gradients
        gradients = -(y_true / y_pred_clipped)
        # Normalize gradients by the number of samples
        gradients /= y_pred.shape[0]
        #print(gradients.shape)  (256, 10)
        return [gradients]
