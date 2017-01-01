"""
Evaluates a model trained by `trainMLP`.

TODO(andrei): Integrate with training script for convenience.
"""
import click

try:
    # Keep this import on top!
    from spacy.en import English
except:
    # Shit, son.
    raise

from os.path import join as pjoin
import sys
import argparse
from keras.models import model_from_json

import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import get_questions_matrix_sum, get_images_matrix, \
    get_answers_matrix
from utils import grouper, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True,
                        help="JSON dump of saved model structure.")
    parser.add_argument('-weights', type=str, required=True,
                        help="Saved weights (checkpoint).")
    parser.add_argument('-results', type=str, required=True,
                        help="File where to write the results.")
    parser.add_argument('-dataroot', type=str, default='/data/vqa')
    args = parser.parse_args()
    root = args.dataroot

    model = model_from_json(open(args.model).read())
    model.load_weights(args.weights)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    questions_val = lines(pjoin(root, 'Preprocessed', 'questions_val2014.txt'))
    answers_val = lines(pjoin(root, 'Preprocessed', 'answers_val2014_all.txt'))
    images_val = lines(pjoin(root, 'Preprocessed', 'images_val2014_all.txt'))
    vgg_model_path = pjoin(root, 'coco', 'vgg_feats.mat')

    print('Model compiled, weights loaded...')
    # TODO(andrei): If this fails, use pickle directly.
    labelencoder = joblib.load(pjoin(root, 'models', 'labelencoder.pkl'))

    features_struct = scipy.io.loadmat(vgg_model_path)
    VGGfeatures = features_struct['feats']
    print('loaded vgg features')
    image_ids = lines(pjoin(root, 'coco_vgg_IDMap.txt'))

    img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        img_map[id_split[0]] = int(id_split[1])

    nlp = English()
    print('loaded word2vec features')

    nb_classes = 1000
    y_predict_text = []
    batchSize = 128

    stuff = list(zip(
        grouper(questions_val, batchSize, fillvalue=questions_val[0]),
        grouper(answers_val, batchSize, fillvalue=answers_val[0]),
        grouper(images_val, batchSize, fillvalue=images_val[0])))

    with click.progressbar(stuff) as pbar:
        for (qu_batch, an_batch, im_batch) in pbar:
            X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
            if 'language_only' in args.model:
                X_batch = X_q_batch
            else:
                X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
                X_batch = np.hstack((X_q_batch, X_i_batch))
            y_predict = model.predict_classes(X_batch, verbose=0)
            y_predict_text.extend(labelencoder.inverse_transform(y_predict))

    correct_val = 0.0
    total = 0
    f1 = open(args.results, 'w')

    for prediction, truth, question, image in zip(y_predict_text, answers_val,
                                                  questions_val, images_val):
        temp_count = 0
        for _truth in truth.split(';'):
            if prediction == _truth:
                temp_count += 1

        if temp_count > 2:
            correct_val += 1
        else:
            correct_val += float(temp_count) / 3

        total += 1
        f1.write(question)
        f1.write('\n')
        f1.write(image)
        f1.write('\n')
        f1.write(prediction)
        f1.write('\n')
        f1.write(truth)
        f1.write('\n')
        f1.write('\n')

    f1.write('Final Accuracy is ' + str(correct_val / total))
    f1.close()

    # TODO(andrei): Re-add this, so we are neat about keeping track of all our
    # results.
    # f1 = open('../results/overall_results.txt', 'a')
    # f1.write(args.weights + '\n')
    # f1.write(str(correct_val / total) + '\n')
    # f1.close()
    print('Final Accuracy on the validation set is', correct_val / total)


if __name__ == "__main__":
    main()
