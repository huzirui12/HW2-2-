import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    TODO:
        - fit
        - call
        - inverse

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        ## TODO: Fetch all the unique labels and create a dictionary with
        ## the unique labels as keys and their one hot encodings as values
        unique_labels = np.unique(data)
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return label_to_index

    def call(self, data):
        ## TODO: Implement call function
        label_to_index = self.fit(data)
        
        one_hot_encoded = np.zeros((len(data), len(label_to_index)))
        for i, label in enumerate(data):
            index = label_to_index[label]
            one_hot_encoded[i, index] = 1
        return one_hot_encoded

    def inverse(self, data):
        ## TODO: Implement inverse function
        labels = []
        for item in data:
            index = np.argmax(item)
            labels.append(self.index_to_label[index])
        return labels