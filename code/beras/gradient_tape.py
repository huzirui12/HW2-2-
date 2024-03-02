from collections import defaultdict
import numpy as np
from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """
        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        
        while queue:
            current_target = queue.pop()
            if id(current_target) not in self.previous_layers:
                continue
            j = grads[id(current_target)]
            layer = self.previous_layers[id(current_target)]
            weight_grads = layer.compose_weight_gradients(j)
            input_gards  =layer.compose_input_gradients(j)
            for w, w_g in zip(layer.weights,weight_grads):
                grads[id(w)] = [w_g]
            for i,i_g in zip(layer.inputs,input_gards):
                grads[id(i)] = [i_g]
            queue.extend(layer.inputs)

        out_grads = [grads[id(source)][0] for source in sources]
        disconnected = [f"var{i}" for i, grad in enumerate(out_grads) if grad is None]

        if disconnected:
            print(f"Warning: The following tensors are disconnected from the target graph: {disconnected}")

        return out_grads