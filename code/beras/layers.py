import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):
    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        return [self.w, self.b]

    def call(self, x: Tensor) -> Tensor:
        return Tensor(x @ self.w + self.b)

    def get_input_gradients(self) -> list[Tensor]:
        return [Tensor(self.w)]

    def get_weight_gradients(self) -> list[Tensor]:
        a = np.expand_dims(self.input_dict['x'], axis=-1)
        b = np.ones(self.b.shape)
        return [Tensor(a),Tensor(b)]


    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
            "xavier uniform",
            "kaiming uniform",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        io_size = (input_size, output_size)
        
        if initializer == "zero":
            w = Variable(np.zeros(io_size))
            b = Variable(np.zeros(output_size))
        elif initializer == "normal":
            w = Variable(np.random.randn(*io_size))
            b = Variable(np.zeros(output_size))
        elif initializer == "xavier":
            w = Variable(np.random.randn(*io_size) * np.sqrt(1 / input_size))
            b = Variable(np.zeros(output_size))
        elif initializer == "kaiming":
            w = Variable(np.random.randn(*io_size) * np.sqrt(2 / input_size))
            b = Variable(np.zeros(output_size))
        elif initializer == "xavier uniform":
            w = Variable(np.random.uniform(-1, 1, io_size) * np.sqrt(6 / input_size))
            b = Variable(np.zeros(output_size))
        elif initializer == "kaiming uniform":
            w = Variable(np.random.uniform(-1, 1, io_size) * np.sqrt(6 / input_size))
            b = Variable(np.zeros(output_size))
        
        return [w, b]
