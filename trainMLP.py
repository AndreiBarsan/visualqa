"""
Baseline from: https://avisingh599.github.io/deeplearning/visual-qa/
In the process of adapting this to our project.
"""

# WARNING: this may cause weird errors when imported after Keras!
import time

try:
    from spacy.en import English
except ImportError as err:
    print("Please make sure you have the `spacy` Python package installed, "
          "that you downloaded its assets (`python -m spacy.en.download`), and "
          "that it is the first thing you import in a Python program.")
    raise


import argparse
import pickle
from os.path import join as pjoin
from random import shuffle
from typing import List

import numpy as np
import scipy.io
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import generic_utils
from sklearn import preprocessing

from features import get_images_matrix, get_answers_matrix
from features import get_questions_matrix_sum
from utils import selectFrequentAnswers, mkdirp, grouper, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_hidden_units', type=int, default=1024)
    parser.add_argument('-num_hidden_layers', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-activation', type=str, default='tanh')
    parser.add_argument('-language_only', type=bool, default=False)
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-model_save_interval', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-dataroot', type=str, default='/data/vqa')
    args = parser.parse_args()

    root = args.dataroot
    q_train_fpath = pjoin(root, 'Preprocessed', 'questions_train2014.txt')
    a_train_fpath = pjoin(root, 'Preprocessed', 'answers_train2014_modal.txt')
    i_train_fpath = pjoin(root, 'Preprocessed', 'images_train2014.txt')
    pretrained_vgg_model_fpath = pjoin(root, 'coco', 'vgg_feats.mat')

    print("Will load Q&A data...")
    questions_train = lines(q_train_fpath)
    answers_train = lines(a_train_fpath)
    # IDs of the images corresponding to the Q&A pairs.
    images_train = lines(i_train_fpath)
    print("Done.")

    # Since we are simplifying the problem of Visual QA to a classification
    # problem in this baseline, we want to limit the number of possible
    # answers, and have the model simply pick the most appropriate one.
    max_answers = 1000
    questions_train, answers_train, images_train = selectFrequentAnswers(
        questions_train, answers_train, images_train, max_answers)

    # encode the remaining answers
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(answers_train)
    nb_classes = len(list(labelencoder.classes_))

    # TODO(andrei): Why are we dumping this?
    mkdirp(pjoin(root, 'models'))
    with open(pjoin(root, 'models', 'labelencoder.pkl'), 'wb') as pfile:
        pickle.dump(labelencoder, pfile)

    print("Loading VGG features...")
    features_struct = scipy.io.loadmat(pretrained_vgg_model_fpath)
    VGGfeatures = features_struct['feats']
    image_ids = lines(pjoin(root, 'coco_vgg_IDMap.txt'))
    print('Loaded vgg features.')

    id_map = {}
    for ids in image_ids:
        id_split = ids.split()
        id_map[id_split[0]] = int(id_split[1])

    print("Loading word2vec data...")
    nlp = English()
    print('loaded word2vec features...')

    # This is the size of the last fully-connected VGG layer, before the
    # softmax.
    img_dim = 4096
    # Standard dimensionality for word2vec embeddings.
    # TODO(andrei): Try GloVe. It should, in theory, work better.
    word_vec_dim = 300

    model = Sequential()
    if args.language_only:
        # Language only means we *ignore the images* and only rely on the
        # question to compute an answers. Interestingly enough, this does not
        # suck horribly.
        model.add(Dense(args.num_hidden_units, input_dim=word_vec_dim,
                        init='uniform'))
    else:
        model.add(Dense(args.num_hidden_units, input_dim=img_dim + word_vec_dim,
                        init='uniform'))

    model.add(Activation(args.activation))
    if args.dropout > 0:
        model.add(Dropout(args.dropout))
    for i in range(args.num_hidden_layers - 1):
        model.add(Dense(args.num_hidden_units, init='uniform'))
        model.add(Activation(args.activation))
        if args.dropout > 0:
            model.add(Dropout(args.dropout))

    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    # Dumps the model info to a file.
    # TODO(andrei): Any particular reason for this dump? Just to ease
    # evaluation later on?
    json_string = model.to_json()
    mkdirp(pjoin(root, 'models'))

    # TODO(andrei): Instead of root, save to current experiment folder instead.
    if args.language_only:
        model_name = 'mlp_language_only_num_hidden_units_{0}' \
                     '_num_hidden_layers_{1}'.format(args.num_hidden_units,
                                                     args.num_hidden_layers)
        model_file_name = pjoin(root, 'models', model_name)
    else:
        model_name = 'mlp_num_hidden_units_{0}' \
                     '_num_hidden_layers_{1}'.format(args.num_hidden_units,
                                                     args.num_hidden_layers)
        model_file_name = pjoin(root, 'models', model_name)
    open(model_file_name + '.json', 'w').write(json_string)

    print('Compiling Keras model...')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Compilation done...')

    print('Training started...')
    for k in range(args.num_epochs):
        epoch_start_ms = int(time.time() * 1000)
        # shuffle the data points before going through them
        index_shuf = list(range(len(questions_train)))
        shuffle(index_shuf)
        questions_train = [questions_train[i] for i in index_shuf]
        answers_train = [answers_train[i] for i in index_shuf]
        images_train = [images_train[i] for i in index_shuf]
        progbar = generic_utils.Progbar(len(questions_train))
        for qu_batch, an_batch, im_batch in zip(
                grouper(questions_train, args.batch_size,
                        fillvalue=questions_train[-1]),
                grouper(answers_train, args.batch_size,
                        fillvalue=answers_train[-1]),
                grouper(images_train, args.batch_size,
                        fillvalue=images_train[-1])):
            X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
            if args.language_only:
                X_batch = X_q_batch
            else:
                X_i_batch = get_images_matrix(im_batch, id_map, VGGfeatures)
                X_batch = np.hstack((X_q_batch, X_i_batch))
            Y_batch = get_answers_matrix(an_batch, labelencoder)
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(args.batch_size, values=[("train loss", loss)])

        epoch_end_ms = int(time.time() * 1000)
        epoch_delta_s = (epoch_end_ms - epoch_start_ms) / 1000.0
        print("Epoch took {0:.1f}s.".format(epoch_delta_s))

        # print type(loss)
        if k % args.model_save_interval == 0:
            model_dump_fname = model_file_name + '_epoch_{:02d}.hdf5'.format(k)
            print('Saving model to file: {0}'.format(model_dump_fname))
            model.save_weights(model_dump_fname)

    model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))


if __name__ == "__main__":
    main()