from keras.layers import Dense, Dropout, Activation, Merge
from keras.models import Sequential

from model import language_models, image_models


class VqaModel(object):
    """
    A multi layer neural network for visual question answering
    """

    def __init__(self, lang_model: language_models.ALanguageModel,
                 img_model: image_models.AImageModel, language_only,
                 num_hidden_units, activation, dropout, num_hidden_layers, nb_classes):
        """
        :param lang_model: the language model to use
        :param img_model: the image model to use
        :param language_only: use a language model only to answer question and ignore images
        :param num_hidden_units: number of hidden units per hidden layer
        :param activation: activation function type
        :param dropout: fraction of nodes which should be dropped out in each training step,
        between 0 and 1.
        :param num_hidden_layers: the number of hidden layers
        :param nb_classes: the number of possible answers we allow (softmax size in the end)
        """
        # Start constructing the Keras model.
        model = Sequential()

        if language_only:
            # Language only means we *ignore the images* and only rely on the
            # question to compute an answers. Interestingly enough, this does not
            # suck horribly.
            model.add(Merge([lang_model.model()], mode='concat', concat_axis=1))
        else:
            model.add(Merge([lang_model.model(), img_model.model()], mode='concat', concat_axis=1))

        if dropout > 0:
            model.add(Dropout(dropout))

        for i in range(num_hidden_layers):
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

    @property
    def model(self):
        return self._model
