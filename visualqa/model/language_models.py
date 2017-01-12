# WARNING: this may cause weird errors when imported after Keras!
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding

try:
    from spacy.en import English
except ImportError as err:
    print("Please make sure you have the `spacy` Python package installed, "
          "that you downloaded its assets (`python -m spacy.en.download`), and "
          "that it is the first thing you import in a Python program.")
    raise

import tensorflow as tf

from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM

from features import get_questions_matrix_sum, get_questions_tensor_timeseries, get_embeddings


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
        print('Loading GloVe data... ', end='', flush=True)
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


class LSTMLanguageModel(ALanguageModel):
    """LSTM language model with word embedding inputs."""
    def __init__(self, lstm_num_layers, lstm_layer_size, trainable_embeddings, **kw):
        """Initializes the Keras LSTM question processing component.

        Args:
            lstm_num_layers: Number of stacked LSTM layers.
            lstm_layer_size: Dimensionality of each LSTM unit.

        Keyword Args:
            max_sentence_length: Maximum number of words to consider in each
                                 question, necessary at train time.
            bidirectional: Whether to use bidirectional LSTM layers.
        """
        print('Loading GloVe data... ', end='', flush=True)
        self._nlp = English()
        with open('/data/vqa/embeddings/glove.42B.300d.txt') as embeddings_file:
            self._nlp.vocab.load_vectors(embeddings_file)
        print('Done.')
        #embedding_dims = 300
        embeddings = get_embeddings(self._nlp.vocab)
        embedding_dims = embeddings.shape[1] 

        # TODO(Bernhard): Investigate how the LSTM parameters influence the
        # overall performance.
        self._max_len = kw.get('max_sentence_length', 15)
        self._bidirectional = kw.get('bidirectional', False)

        self._model = Sequential()
        shallow = lstm_num_layers == 1  # marks a one layer LSTM

        if trainable_embeddings:
            # if embeddings are trainable we have to enforce CPU usage in order to not run out of memory.
            # this is device dependent.
            # TODO(Bernhard): preprocess questions ans vocab and try if we can get rid of enough words to make
            # this run on gpu anyway
            with tf.device("/cpu:0"):
                self._model.add(Embedding(embeddings.shape[0], embeddings.shape[1],
                                      input_length=self._max_len, trainable=True, weights=[embeddings]))
        else:
            # a non-trainable embedding layer can run on GPU without exhausting all the memory
            self._model.add(Embedding(embeddings.shape[0], embeddings.shape[1],
                                      input_length=self._max_len, trainable=False, weights=[embeddings]))

        lstm = LSTM(output_dim=lstm_layer_size,
                    return_sequences=not shallow,
                    input_shape=(self._max_len, embedding_dims))
        if self._bidirectional:
            lstm = Bidirectional(lstm)
        self._model.add(lstm)
        if not shallow:
            for i in range(lstm_num_layers-2):
                lstm = LSTM(output_dim=lstm_layer_size, return_sequences=True)
                if self._bidirectional:
                    lstm = Bidirectional(lstm)
                self._model.add(lstm)

            lstm = LSTM(output_dim=lstm_layer_size, return_sequences=False)
            if self._bidirectional:
                lstm = Bidirectional(lstm)
            self._model.add(lstm)

    def model(self):
        return self._model

    def process_input(self, question):
        return get_questions_tensor_timeseries(question, self._nlp, self._max_len)
