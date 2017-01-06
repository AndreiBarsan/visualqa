from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential


class VqaModel(object):
    """
    A multi layer neural network for visual question answering
    """

    def __init__(self, language_only, num_hidden_units, word_vec_dim,
                 activation, dropout, num_hidden_layers, nb_classes):
        """
        :param language_only: use a language model only to answer question and ignore images
        :param num_hidden_units: number of hidden units per hidden layer
        :param word_vec_dim: dimensionality of the word embedding vector
        :param activation: activation function type
        :param dropout: fraction of nodes which should be dropped out in each training step,
        between 0 and 1.
        :param num_hidden_layers: the number of hidden layers
        :param nb_classes: the number of possible answers we allow (softmax size in the end)
        """

        # TODO(Bernhard): shouldn't word_vec_dim be question_vec_dim? It is already the result
        # of the entire question and not only a single word? Dimension of question vectors
        # do not have to equal dimension of the word embeddings.

        # This is the size of the last fully-connected VGG layer, before the
        # softmax.
        img_dim = 4096

        # Start constructing the Keras model.
        model = Sequential()
        if language_only:
            # Language only means we *ignore the images* and only rely on the
            # question to compute an answers. Interestingly enough, this does not
            # suck horribly.
            model.add(Dense(num_hidden_units, input_dim=word_vec_dim,
                            init='uniform'))
        else:
            model.add(Dense(num_hidden_units, input_dim=img_dim + word_vec_dim,
                            init='uniform'))

        model.add(Activation(activation))

        if dropout > 0:
            model.add(Dropout(dropout))

        for i in range(num_hidden_layers - 1):
            model.add(Dense(num_hidden_units, init='uniform'))
            model.add(Activation(activation))
            if dropout > 0:
                model.add(Dropout(dropout))

        model.add(Dense(nb_classes, init='uniform'))
        model.add(Activation('softmax'))

        print('Compiling Keras model...')
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        print('Compilation done...')
        self._model = model

    def getmodel(self):
        return self._model
