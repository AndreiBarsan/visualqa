from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers.core import Reshape

from features import get_questions_matrix_sum


class ALanguageModel(ABC):
    """
    Abstract base class for a language model. Inherit this class
    and implement the abstract methods to define a new language
    model. The model is then combined with the graphical model to
    operate on the VQA data set.
    """

    @abstractmethod
    def model(self):
        """
        :return: the *uncompiled* language model
        """
        pass

    @abstractmethod
    def process_input(self, question, embeddings):
        """
        Processing the input is model specific. While an easy
        model would just sum up embedding vectors, more advanced
        models might use a LSTM layer. This method is called in training
        and testing and should return the input vector for the neural
        network. for a given question and the corresponding word embeddings.
        :param question: a list of unicode objects
        :param embeddings: instance of the class English() from spacy.en
        :return: the input vector for the language model
        """
        # TODO(Bernhard) make this independent of spacy.en
        pass


class SumUpLanguageModel(ALanguageModel):
    """
    Easiest language model: sums up all the word embeddings.
    """

    def __init__(self, embedding_dims):
        self._model = Sequential()
        self._model.add(Reshape(input_shape=(embedding_dims,), target_shape=(embedding_dims,)))

    def model(self):
        return self._model

    def process_input(self, question, embeddings):
        return get_questions_matrix_sum(question, embeddings)
