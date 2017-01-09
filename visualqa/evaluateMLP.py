"""
Evaluates a model trained by `trainMLP`.

TODO(andrei): Integrate with training script for convenience.
"""
try:
    # Keep this import on top!
    from spacy.en import English
except:
    # Shit, son.
    raise

import click
from os.path import join as pjoin
import argparse
from keras.models import model_from_json

import scipy.io
from sklearn.externals import joblib

from features import get_questions_matrix_sum, get_images_matrix, \
    get_answers_matrix, get_questions_tensor_timeseries
from utils import grouper, lines, batchify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True,
                        help="JSON dump of saved model structure.")
    parser.add_argument('-weights', type=str, required=True,
                        help="Saved weights (checkpoint).")
    parser.add_argument('-results', type=str, required=True,
                        help="File where to write the results.")
    parser.add_argument('-results_json', type=str, required=True,
                        help="File where to dump the evaluation results in "
                             "JSON format, so that the official VQA toolkit "
                             "can read it.")
    parser.add_argument('-dataroot', type=str, default='/data/vqa')
    args = parser.parse_args()
    root = args.dataroot

    model = model_from_json(open(args.model).read())
    model.load_weights(args.weights)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    questions_val = lines(pjoin(root, 'Preprocessed', 'questions_val2014.txt'))
    questions_id = lines(pjoin(root, 'Preprocessed', 'questions_id_val2014.txt'))
    answers_val = lines(pjoin(root, 'Preprocessed', 'answers_val2014_all.txt'))
    images_val = lines(pjoin(root, 'Preprocessed', 'images_val2014_all.txt'))
    vgg_model_path = pjoin(root, 'coco', 'vgg_feats.mat')

    print('Model compiled, weights loaded...')

    # Load the encoder which converts answers to IDs, saved in the same
    # folder as the rest of the dumps.
    exp_root = args.weights[:args.weights.rfind('/')]
    labelencoder = joblib.load(pjoin(exp_root, 'labelencoder.pkl'))

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

    # TODO(andrei): Configure this via args.
    batchSize = 512

    stuff = batchify(batchSize, questions_val, answers_val, images_val)
    with click.progressbar(stuff) as pbar:
        for (qu_batch, an_batch, im_batch) in pbar:
            # TODO(Bernhard): make this choose the right preprocessing and right model,
            # for now you have to plug it in manually
            #X_q_batch = get_questions_matrix_sum(qu_batch, nlp) # for sum up model
            X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, 20) # for LSTM model

            if 'language_only' in args.model:
                y_predict = model.predict_classes([X_q_batch], verbose=0)
            else:
                X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
                y_predict = model.predict_classes([X_q_batch, X_i_batch], verbose=0)
            # TODO(Bernhard): verify that predict_classes sets dropout to 0
            y_predict_text.extend(labelencoder.inverse_transform(y_predict))

    correct_val = 0.0
    total = 0
    f1 = open(args.results, 'w')
    print("Will dump resulting answers in JSON format to file: [{0}]".format(
        args.results_json
    ))
    result_file_json = open(args.results_json, 'w')
    result_file_json.write("[")

    all_preds = list(zip(y_predict_text, answers_val, questions_val,
                        questions_id, images_val))
    for idx, (prediction, truth, question, question_id, image) in enumerate(all_preds):
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

        # Note: Double-braces are escaped braces in Python format strings.
        result_file_json.write(
            '{{"answer": "{0}", "question_id": {1}}}{2}\n'.format(
                prediction, question_id, ',' if idx < len(all_preds) - 1 else ''))

    result_file_json.write("]\n")
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
