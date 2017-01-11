"""Utilities for going from raw data to NN-friendly formats.

Source: https://github.com/avisingh599/visual-qa
"""

import numpy as np
from keras.utils import np_utils


def get_questions_tensor_timeseries(questions, nlp, max_length):
    '''
    Returns a time series of word vectors for tokens in the question

    Input:
    questions: list of unicode objects
    nlp: an instance of the class English() from spacy.en
    max_lenght: the maximum number of words in each question to take into account

    Output:
    A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
    '''
    # assert not isinstance(questions, basestring)
    questions_tensor = np.zeros((len(questions), max_length), dtype='int32')
    concatenated = 0
    for i, doc in enumerate(questions):
        j = 0
        for token in nlp(doc):
            if j < max_length and token.has_vector:
                questions_tensor[i, j] = token.rank
                j += 1
            else:
                if j == max_length:
                    concatenated += 1
                    break
    if concatenated > 0:
        print("warning {0} questions concatenated".format(concatenated))
    return questions_tensor


def get_questions_matrix_sum(questions, nlp):
    """Sums the word vectors of all the tokens in a question.

    Args
        questions: list of unicode objects
        nlp: an instance of the class English() from spacy.en

    Returns
        A numpy array of shape: (nb_samples, word_vec_dim)
    """
    # assert not isinstance(questions, basestring)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_matrix = np.zeros((nb_samples, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            questions_matrix[i, :] += tokens[j].vector

    return questions_matrix


def get_answers_matrix(answers, encoder):
    '''Converts string objects to class labels

    Input:
    answers:	a list of unicode objects
    encoder:	a scikit-learn LabelEncoder object

    Output:
    A numpy array of shape (nb_samples, nb_classes)
    '''
    # assert not isinstance(answers, basestring)
    y = encoder.transform(answers)  # string to numerical class
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y


def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    '''
    Gets the 4096-dimensional CNN features for the given COCO
    images

    Input:
    img_coco_ids: 	A list of strings, each string corresponding to
                      the MS COCO Id of the relevant image
    img_map: 		A dictionary that maps the COCO Ids to their indexes
                    in the pre-computed VGG features matrix
    VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)

    Ouput:
    A numpy matrix of size (nb_samples, nb_dimensions)
    '''
    assert not isinstance(img_coco_ids, str)
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j, :] = VGGfeatures[:, img_map[img_coco_ids[j]]]

    return image_matrix


def get_embeddings(vocab):
    '''
    Extracts word embeddings from a nlp object (Spacy)

    Input:
    vocab: nlp.vocab
    Ouput:
    A numpy array containing the word embeddings
    '''
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.zeros((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors