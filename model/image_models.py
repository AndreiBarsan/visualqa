from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers.core import Reshape

from features import get_images_matrix


class AImageModel(ABC):
    """
    Abstract base class for an image model. Inherit this class
    and implement the abstract methods to define a new model on the
    input images. The model is then combined with the language model to
    operate on the VQA data set.
    """

    @abstractmethod
    def model(self):
        """
        :return: the *uncompiled* image model
        """
        pass

    def process_input(self, image_features):
        """
        Processing the input is model specific. This method is called
        in training and testing and should return the input to the neural
        net for a given image.
        :param image_features: whatever is needed to compute the input
        :return: the input vector for the image model
        """
        pass


class VGGImageModel(AImageModel):
    """
    This model uses the last layer (before softmax) of the VGG net
    as image features. The convolutions are not caluclated but pre-computed
    image features are looked up and directly plugged into the network.
    """
    def __init__(self):
        # Dimensionality of image features
        img_dim = 4096
        self._model = Sequential()
        self._model.add(Reshape(input_shape=(img_dim,), target_shape=(img_dim,)))

    def model(self):
        return self._model

    def process_input(self, image_features):
        """
        :param a triple:
        img_coco_ids: 	A list of strings, each string corresponding to
                        the MS COCO Id of the relevant image
        img_map: 		A dictionary that maps the COCO Ids to their indexes
                        in the pre-computed VGG features matrix
        VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)
        :return: A numpy matrix of size (nb_samples, nb_dimensions)
        """
        (img_coco_ids, img_map, vgg_features) = image_features
        return get_images_matrix(img_coco_ids, img_map, vgg_features)
