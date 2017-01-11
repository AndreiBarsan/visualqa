"""
Baseline from: https://avisingh599.github.io/deeplearning/visual-qa/
In the process of adapting this to our project.
"""

import time
import argparse
import pickle
import random
from os.path import join as pjoin
from random import shuffle

import keras
import numpy as np
from keras.utils import generic_utils
from sklearn import preprocessing
from sklearn.utils import shuffle as sklearn_shuffle
import tensorflow as tf

# TODO(andrei): Use proper module imports.
from config import RANDOM_SEED
from features import get_answers_matrix
from model import vqa_model, language_models, image_models
from utils import *



def load_train_data(data_root):
    """
    load the training data and return the loaded questions, answers and images
    """
    q_train_fpath = pjoin(data_root, 'Preprocessed', 'questions_train2014.txt')
    a_train_fpath = pjoin(data_root, 'Preprocessed', 'answers_train2014_modal.txt')
    i_train_fpath = pjoin(data_root, 'Preprocessed', 'images_train2014.txt')

    print("Will load Q&A data...")
    questions_train = lines(q_train_fpath)
    answers_train = lines(a_train_fpath)
    # IDs of the images corresponding to the Q&A pairs.
    images_train = lines(i_train_fpath)
    print("Done.")

    return questions_train, answers_train, images_train


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_hidden_units', type=int, default=1024,
                        help="Hidden layer size in the final MLP.")
    parser.add_argument('-num_hidden_layers', type=int, default=3,
                        help="Number of hidden layers in the final MLP.")
    parser.add_argument('-lstm_num_layers', type=int, default=1)
    parser.add_argument('-lstm_layer_size', type=int, default=256)
    parser.add_argument('-trainable_embeddings', type=bool, default=False)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-activation', type=str, default='relu')
    parser.add_argument('-language_only', type=bool, default=False)
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-model_save_interval', type=int, default=5)
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
    parser.add_argument('-valid_ratio', type=float, default=0.05,
                        help="How much of the training data to use for "
                             "validation during training.")
    # get args
    args = parser.parse_args()
    # Dump the arguments so we know which parameters we used for training.
    with open(pjoin(args.experiment_root, 'args.pkl'), 'wb') as pfile:
        pickle.dump(args, pfile)
    return args


def construct_model(args, data_root, experiment_root, nb_classes):
    """Constructs the final model to use in training."""

    # specify language model:
    lang_model = language_models.LSTMLanguageModel(args.lstm_num_layers,
                                                   args.lstm_layer_size,
                                                   args.trainable_embeddings)

    # specify image mode:
    img_model = image_models.VGGImageModel(data_root)

    # specify vqa mode:
    final_model = vqa_model.VqaModel(lang_model, img_model,
                                     args.language_only, args.num_hidden_units,
                                     args.activation, args.dropout,
                                     args.num_hidden_layers, nb_classes)

    # Dump the model structure so we can use it later (we dump just the raw
    # weights with every checkpoint).
    json_string = final_model.model.to_json()
    model_file_name = pjoin(experiment_root, 'model.json')
    open(model_file_name, 'w').write(json_string)

    return final_model, lang_model, img_model


def main():
    args = parse_arguments()

    data_root = args.dataroot
    experiment_root = args.experiment_root

    # Set both the numpy and the Python random seeds.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # Load data needed for training and save all parameters/mappings to make
    # sure experiments are reproducible
    questions_train_all, answers_train_all, images_train_all = load_train_data(
        data_root)

    # Since we are simplifying the problem of Visual QA to a classification
    # problem in this baseline, we want to limit the number of possible
    # answers, and have the model simply pick the most appropriate one.
    max_answers = 1000
    questions_train_all, answers_train_all, images_train_all = \
        select_frequent_answers(questions_train_all, answers_train_all,
                                images_train_all, max_answers)

    # Encode the remaining (top max_answers) answers and save the mapping.
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(answers_train_all)
    nb_classes = len(list(labelencoder.classes_))
    with open(pjoin(experiment_root, 'labelencoder.pkl'), 'wb') as pfile:
        pickle.dump(labelencoder, pfile)

    # The initial shuffle ensures that the train-val split is randomized
    # depending on the random seed, and not fixed every time (which would be
    # very bad).
    print("Performing initial shuffle...")
    questions_train_all, answers_train_all, images_train_all = sklearn_shuffle(
        questions_train_all, answers_train_all, images_train_all
    )

    train_all_count = len(questions_train_all)
    valid_count = int(train_all_count * args.valid_ratio)
    train_count = train_all_count - valid_count

    print("We have {0} total Q-A pairs. Will use {1:.2f}% for validation, "
          "which is {2} data points. {3} data points will be used for "
          "actual training.".format(train_all_count, args.valid_ratio * 100.0,
                                    valid_count, train_count))

    questions_train = questions_train_all[:train_count]
    answers_train = answers_train_all[:train_count]
    images_train = images_train_all[:train_count]
    # Note again that this is NOT the official validation set, but just a
    # fraction (`args.valid_ratio`) of the training set. The full validation
    # set evaluation is performed separately.
    questions_valid = questions_train_all[train_count:]
    answers_valid = answers_train_all[train_count:]
    images_valid = images_train_all[train_count:]

    # construct the model
    final_model, lang_model, img_model = construct_model(args, data_root, experiment_root, nb_classes)
    model = final_model.model

    # Compute val error K times per epoch.
    val_per_epoch = 4
    eval_valid_every = int((train_count / args.batch_size) / val_per_epoch)

    # Perform Tensorboard-friendly dumps.
    # TODO(andrei): This only works when using Keras's 'fit' method directly.
    # tensorboard_log_dir = pjoin(experiment_root, 'logs')
    # tensorboard_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,
    #                                              histogram_freq=0,
    #                                              write_graph=True,
    #                                              write_images=False)

    # The training part starts here
    print('Training started...')
    last_valid_loss = 10
    for epoch in range(args.num_epochs):
        epoch_start_ms = int(time.time() * 1000)
        # shuffle the data points before going through them
        questions_train, answers_train, images_train = sklearn_shuffle(
            questions_train, answers_train, images_train
        )
        progbar = generic_utils.Progbar(len(questions_train))
        batches = batchify(args.batch_size, questions_train, answers_train,
                           images_train)
        for batch_idx, (qu_batch, an_batch, im_batch) in enumerate(batches):
            # Extract batch vectors to train on
            # Converts the answers to their index (we're just doing
            # classification at this point)
            y_batch = get_answers_matrix(an_batch, labelencoder)

            # train on language only or language and image both
            if args.language_only:
                x_q_batch = lang_model.process_input(qu_batch)
                loss = model.train_on_batch(x_q_batch, y_batch)
            else:
                x_q_batch = lang_model.process_input(qu_batch)
                x_i_batch = img_model.process_input(im_batch)
                loss = model.train_on_batch([x_q_batch, x_i_batch], y_batch)

            if (batch_idx + 1) % eval_valid_every == 0:
                # It's time to validate on the held-out part of the training
                # dataset.
                batch_val_losses = []
                val_batches = batchify(args.batch_size, questions_valid,
                                       answers_valid, images_valid)
                for (qu_val_batch, an_val_batch, im_val_batch) in val_batches:
                    y_val_batch = get_answers_matrix(an_val_batch, labelencoder)
                    if args.language_only:
                        val_loss = model.test_on_batch(
                            lang_model.process_input(qu_val_batch),
                            y_val_batch)
                    else:
                        val_loss = model.test_on_batch([
                            lang_model.process_input(qu_val_batch),
                            img_model.process_input(im_val_batch)
                        ], y_val_batch)

                    batch_val_losses.append(val_loss)

                # The validation loss is just the average of the individual
                # losses computed for each batch of the validation data.
                last_valid_loss = np.mean(batch_val_losses)

            # if batch_idx % progress_update_every == 0:
            # Important: because of retarded reasons, the progress bar
            # averages these values, so the reported validation loss will
            # have a bit of lag.
            progbar.add(args.batch_size,
                        values=[("tra-loss", loss), ("val-loss", last_valid_loss)])

        epoch_end_ms = int(time.time() * 1000)
        epoch_delta_s = (epoch_end_ms - epoch_start_ms) / 1000.0
        print("Epoch {0}/{1} took {2:.1f}s.".format(
            (epoch + 1), args.num_epochs, epoch_delta_s))
        print("Latest validation loss: {0:4f}".format(last_valid_loss))

        # Dump a checkpoint periodically.
        if (epoch + 1) % args.model_save_interval == 0:
            model_dump_fname = pjoin(experiment_root,
                                     'weights_{0}.hdf5'.format(epoch + 1))
            print('Saving model to file: {0}'.format(model_dump_fname))
            model.save_weights(model_dump_fname)

        # Compute overall accuracy periodically on OFFICIAL full validation
        # set (but not too often, as it can get quite slow).
        if (epoch + 1) % args.model_eval_full_valid_interval == 0:
            # TODO(andrei): Implement this in a neat way.
            pass

    # TODO(Bernhard): catch control+c and store last parameters...
    # Final checkpoint dump.
    model.save_weights(pjoin(experiment_root, 'weights_{0}_final.hdf5'.format(
        epoch + 1)))


if __name__ == "__main__":
    main()
