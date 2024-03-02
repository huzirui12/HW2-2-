import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    """
    TODO: Implement for default intermediate activation.
        - call function
        - input gradients
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def call(self, x) -> Tensor:
        """TODO: Leaky ReLu forward propagation! """
        result = np.maximum(x, self.alpha * x)
        return Tensor(result)

    def get_input_gradients(self) -> list[Tensor]:
        """
        TODO: Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet!
        Make sure not to mutate any instance variables. Return a NEW list[tensor(s)]
        """
        grad = np.ones_like(self.inputs[0], dtype=np.float32)
        grad[self.inputs[0] <= 0] = self.alpha
        gradient_tensor = Tensor(grad)
        return [gradient_tensor]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    """
    TODO: Implement for default output activation to bind output to 0-1
        - call function
        - input_gradients 
    """ 

    def call(self, x) -> Tensor:  
        self.x=x
        outputs = 1 / (1 + np.exp(-x))
        return Tensor(outputs)


    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet!
        Make sure not to mutate any instance variables. Return a NEW list[tensor(s)]
         """
        out=np.array(1 / (1 + np.exp(-self.x)))
        sigmoid_grad = out*(1-out)
        return [Tensor(sigmoid_grad)]


    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    """
    TODO: Implement for default output activation to bind output to 0-1
        - call function
        - input_gradients
    """

    def call(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        self.x = x
        z = x - np.max(x, axis=-1, keepdims=True)  # Subtract max for numerical stability
        exps = np.exp(z)
        outputs = exps / np.sum(exps, axis=-1, keepdims=True)
        return Tensor(outputs)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        # https://stackoverflow.com/questions/48633288/how-to-assign-elements-into-the-diagonal-of-a-3d-matrix-efficiently
       

        n = self.x
        z = n - np.max(n, axis=-1, keepdims=True)  # Subtract max for numerical stability
        exps = np.exp(z)
        outputs = exps / np.sum(exps, axis=-1, keepdims=True)
        
        # Initialize the Jacobian matrix for each instance in the batch
        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]
        jacobian_matrices = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        jacobian_matrices[i, j, k] = outputs[i, j] * (1 - outputs[i, j])
                    else:
                        jacobian_matrices[i, j, k] = -outputs[i, j] * outputs[i, k]
        
        return [Tensor(jacobian_matrices)]
