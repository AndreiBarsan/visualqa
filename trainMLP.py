"""
Baseline from: https://avisingh599.github.io/deeplearning/visual-qa/
In the process of adapting this to our project.
"""

# WARNING: this may cause weird errors when imported after Keras!
try:
    from spacy.en import English
except ImportError as err:
    print("Please make sure you have the `spacy` Python package installed, "
          "that you downloaded its assets (`python -m spacy.en.download`), and "
          "that it is the first thing you import in a Python program.")
    raise

import time
import argparse
import pickle
from os.path import join as pjoin
from random import shuffle

import scipy.io
from keras.utils import generic_utils
from sklearn import preprocessing

from features import get_answers_matrix
from utils import select_frequent_answers, grouper, lines

from model import vqa_model, language_models, image_models


def construct_models(args, nb_classes):
    # Standard dimensionality for word2vec embeddings
    word_vec_dim = 300

    # Dimensionality of image features
    img_dim = 4096

    # specify language model:
    lang_model = language_models.SumUpLanguageModel(word_vec_dim)

    # specify image mode:
    img_model = image_models.VGGImageModel(img_dim)

    # specify vqa mode:
    final_model = vqa_model.VqaModel(lang_model, img_model,
                                  args.language_only, args.num_hidden_units,
                                  args.activation, args.dropout,
                                  args.num_hidden_layers, nb_classes)
    return [final_model, lang_model, img_model]


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
    parser.add_argument('-experiment_root', type=str, default='.',
                        help="Folder where things such as checkpoints are "
                             "saved. By default, this is the program's "
                             "working directory, since the Fabric deployment "
                             "script creates a custom work directory for "
                             "training every time.")
    parser.add_argument('-model_eval_full_valid_interval', type=str,
                        default=20, help="After how many epochs to run a full "
                                         "validation-set evaluation.")
    args = parser.parse_args()

    data_root = args.dataroot
    experiment_root = args.experiment_root

    # Load data needed for training and save all parameters/mappings to make
    # sure experiments are reproducible

    q_train_fpath = pjoin(data_root, 'Preprocessed', 'questions_train2014.txt')
    a_train_fpath = pjoin(data_root, 'Preprocessed', 'answers_train2014_modal.txt')
    i_train_fpath = pjoin(data_root, 'Preprocessed', 'images_train2014.txt')
    pretrained_vgg_model_fpath = pjoin(data_root, 'coco', 'vgg_feats.mat')

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
    questions_train, answers_train, images_train = select_frequent_answers(
        questions_train, answers_train, images_train, max_answers)

    print("Loading VGG features...")
    features_struct = scipy.io.loadmat(pretrained_vgg_model_fpath)
    vgg_features = features_struct['feats']
    image_ids = lines(pjoin(data_root, 'coco_vgg_IDMap.txt'))
    print('Loaded vgg features.')

    # Encode the remaining (top max_answers) answers and save the mapping.
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(answers_train)
    nb_classes = len(list(labelencoder.classes_))
    with open(pjoin(experiment_root, 'labelencoder.pkl'), 'wb') as pfile:
        pickle.dump(labelencoder, pfile)

    # Dump the arguments so we know which parameters we used for training.
    with open(pjoin(experiment_root, 'args.pkl'), 'wb') as pfile:
        pickle.dump(args, pfile)

    id_map = {}
    for ids in image_ids:
        id_split = ids.split()
        id_map[id_split[0]] = int(id_split[1])

    print("Loading word2vec data...")
    # TODO(andrei): Try GloVe. It should, in theory, work better. The spacy
    # library may support them, and if not, we can always do it manually.
    nlp = English()
    print('Loaded word2vec features.')

    # construct the models
    final_model, lang_model, img_model = construct_models(args, nb_classes)
    model = final_model.get_model()

    # Dump the model structure so we can use it later (we dump just the raw
    # weights with every checkpoint).
    json_string = model.to_json()
    model_file_name = pjoin(experiment_root, 'model.json')
    open(model_file_name, 'w').write(json_string)

    # The training part starts here

    # TODO(andrei): If possible, pre-compute sums of all questions and encode
    # all answers in advance.
    # TODO(andrei): Tensorboard. Keras has support for it!
    print('Training started...')
    for epoch in range(args.num_epochs):
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

            # Extract batch vectors to train on

            # Converts the answers to their index (we're just doing
            # classification at this point)
            y_batch = get_answers_matrix(an_batch, labelencoder)

            # train on language only or language and image both
            if args.language_only:
                x_q_batch = lang_model.process_input(qu_batch, nlp)
                loss = model.train_on_batch(x_q_batch, y_batch)
            else:
                x_q_batch = lang_model.process_input(qu_batch, nlp)
                x_i_batch = img_model.process_input((im_batch, id_map, vgg_features))
                loss = model.train_on_batch([x_q_batch, x_i_batch], y_batch)

            progbar.add(args.batch_size, values=[("train loss", loss)])

        epoch_end_ms = int(time.time() * 1000)
        epoch_delta_s = (epoch_end_ms - epoch_start_ms) / 1000.0
        print("Epoch took {0:.1f}s.".format(epoch_delta_s))

        # Dump a checkpoint periodically.
        if epoch % args.model_save_interval == 0:
            model_dump_fname = pjoin(experiment_root, 'weights_{0}.hdf5'.format(epoch))
            print('Saving model to file: {0}'.format(model_dump_fname))
            model.save_weights(model_dump_fname)

        # Compute overall accuracy periodically (but not too often, as it can
        # get quite slow).
        if (epoch + 1) % args.model_eval_full_valid_interval == 0:
            # TODO(andrei): Implement this in a neat way.
            pass

    # Final checkpoint dump.
    model.save_weights(pjoin(experiment_root, 'weights_{0}.hdf5'.format(epoch)))


if __name__ == "__main__":
    main()
