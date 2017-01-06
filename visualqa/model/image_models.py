from abc import ABC, abstractmethod

from keras.models import Sequential
from keras.layers.core import Reshape

from features import get_images_matrix

from os.path import join as pjoin

from utils import lines

import scipy.io


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

    def process_input(self, image_batch):
        """
        Processing the input is model specific. This method is called
        in training and testing and should return the input to the neural
        net for a given image batch.
        :param image_batch: the batch of images
        :return: the input vector for the image model
        """
        pass


class VGGImageModel(AImageModel):
    """
    This model uses the last layer (before softmax) of the VGG net
    as image features. The convolutions are not caluclated but pre-computed
    image features are looked up and directly plugged into the network.
    """
    def __init__(self, data_root):
        # Dimensionality of image features
        img_dim = 4096
        self._model = Sequential()
        self._model.add(Reshape(input_shape=(img_dim,), target_shape=(img_dim,)))

        # Load the precomputed VGG features
        print("Loading VGG features...")
        pretrained_vgg_model_fpath = pjoin(data_root, 'coco', 'vgg_feats.mat')
        features_struct = scipy.io.loadmat(pretrained_vgg_model_fpath)
        self._vgg_features = features_struct['feats']
        image_ids = lines(pjoin(data_root, 'coco_vgg_IDMap.txt'))
        print ("Done.")

        self._id_map = {}
        for ids in image_ids:
            id_split = ids.split()
            self._id_map[id_split[0]] = int(id_split[1])

    def model(self):
        return self._model

    def process_input(self, image_batch):
        """
        :param a triple:
        img_coco_ids: 	A list of strings, each string corresponding to
                        the MS COCO Id of the relevant image
        img_map: 		A dictionary that maps the COCO Ids to their indexes
                        in the pre-computed VGG features matrix
        VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)
        :return: A numpy matrix of size (nb_samples, nb_dimensions)
        """
        return get_images_matrix(image_batch, self._id_map, self._vgg_features)
