# WARNING: this may cause weird errors when imported after Keras!
try:
    from spacy.en import English
except ImportError as err:
    print("Please make sure you have the `spacy` Python package installed, "
          "that you downloaded its assets (`python -m spacy.en.download`), and "
          "that it is the first thing you import in a Python program.")
    raise

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
    def process_input(self, question):
        """
        Processing the input is model specific. While an easy
        model would just sum up embedding vectors, more advanced
        models might use a LSTM layer. This method is called in training
        and testing and should return the input vector for the neural
        network for a given question.
        :param question: a list of unicode objects
        :return: the input vector for the language model
        """
        pass


class SumUpLanguageModel(ALanguageModel):
    """
    Easiest language model: sums up all the word embeddings.
    """

    def __init__(self):
        print('Loading glove data...')
        self._nlp = English()
        # TODO(Bernhard): try word2vec instead of glove..
        print('Done.')

        # embedding_dims of glove
        embedding_dims = 300

        self._model = Sequential()
        self._model.add(Reshape(input_shape=(embedding_dims,), target_shape=(embedding_dims,)))

    def model(self):
        return self._model

    def process_input(self, question):
        return get_questions_matrix_sum(question, self._nlp)
