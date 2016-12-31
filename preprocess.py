#!/usr/bin/env python3
"""Tools for data pre-processing.

Adapted from: https://github.com/avisingh599/visual-qa/blob/master/scripts/dumpText.py
"""

import operator
import argparse
import json
from os.path import join as pjoin
from typing import Dict

from spacy.en import English
import click

# TODO(andrei): Type annotations, documentation.


def get_modal_answer(answers):
    candidates = {}
    for i in range(10):
        candidates[answers[i]['answer']] = 1

    for i in range(10):
        candidates[answers[i]['answer']] += 1

    return max(candidates.items(), key=operator.itemgetter(1))[0]


def get_all_answer(answers):
    answer_list = []
    for i in range(10):
        answer_list.append(answers[i]['answer'])

    return ';'.join(answer_list)


def pretty(d: Dict) -> str:
    """Pretty print a dictionary object."""
    res = "{\n"
    for (key, val) in d.items():
        res += "\t{0}: {1},\n".format(key, val)

    res += "}"
    return res


def main():
    """Extracts the useful things from the JSON files into multiple text files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, default='train',
                        help='Specify which part of the dataset you want to dump to text. Your options are: train, val, test, test-dev')
    parser.add_argument('-answers', type=str, default='modal',
                        help='Specify if you want to dump just the most frequent answer for each questions (modal), or all the answers (all)')
    args = parser.parse_args()

    nlp = English()  # used for conting number of tokens

    # TODO(andrei): Pass as argument after converting to click. And do not
    # hardcode local paths at all.
    root = pjoin('/Volumes', 'palos', 'Datasets', 'vqa')

    fname_map = {
        'train': {
            'annotations': pjoin('Annotations', 'mscoco_train2014_annotations.json'),
            'questions': pjoin('Questions', 'OpenEnded_mscoco_train2014_questions.json'),
            'questions_out': pjoin('Preprocessed', 'questions_train2014.txt'),
            'questions_id_out': pjoin('Preprocessed', 'questions_id_train2014.txt'),
            'questions_lengths_out': pjoin('Preprocessed', 'questions_lengths_train2014.txt'),
            'answers_file_out': pjoin('Preprocessed', 'answers_train2014_{0}.txt'.format(args.answers)),
            'coco_image_id_out': pjoin('Preprocessed', 'images_train2014.txt'),
            'data_split': 'training data',
        },
        'val': {
            'annotations': pjoin('Annotations', 'mscoco_val2014_annotations.json'),
            'questions': pjoin('Questions', 'OpenEnded_mscoco_val2014_questions.json'),
            'questions_out': pjoin('Preprocessed', 'questions_val2014.txt'),
            'questions_id_out': pjoin('Preprocessed', 'questions_id_val2014.txt'),
            'questions_lengths_out': pjoin('Preprocessed', 'questions_lengths_val2014.txt'),
            'answers_file_out': pjoin('Preprocessed', 'answers_val2014_{0}.txt'.format(args.answers)),
            # TODO(andrei): Does the 'all' prefix make sense here?
            'coco_image_id_out': pjoin('Preprocessed', 'images_val2014_all.txt'),
            'data_split': 'validation data',
        },
        'test-dev': {
            'questions': pjoin('Questions', 'OpenEnded_mscoco_test-dev2015_questions.json'),
            'questions_out': pjoin('Preprocessed', 'questions_test-dev2015.txt'),
            'questions_id_out': pjoin('Preprocessed', 'questions_id_test-dev2015.txt'),
            'questions_lengths_out': pjoin('Preprocessed', 'questions_lengths_test-dev2015.txt'),
            'coco_image_id_out': pjoin('Preprocessed', 'images_test-dev2015.txt'),
            'data_split': 'test-dev data',
        },
        'test': {
            'questions': pjoin('Questions', 'OpenEnded_mscoco_test2015_questions.json'),
            'questions_out': pjoin('Preprocessed', 'questions_test2015.txt'),
            'questions_id_out': pjoin('Preprocessed', 'questions_id_test2015.txt'),
            'questions_lengths_out': pjoin('Preprocessed', 'questions_lengths_test2015.txt'),
            'coco_image_id_out': pjoin('Preprocessed', 'images_test2015.txt'),
            'data_split': 'test data',
        }
    }

    # Prefix all the paths with the name of the root folder.
    fname_map = {fname_key: {k: pjoin(root, path) if k != 'data_split' else path
                             for (k, path) in fname_map[fname_key].items()}
                 for fname_key in fname_map}

    if args.split not in fname_map:
        raise RuntimeError(
            'Incorrect split. Available choices are:\ntrain\nval\ntest-dev\ntest')

    fnames = fname_map[args.split]
    question_fname = fnames['questions']
    annotation_fname = fnames['annotations']
    questions_file = open(fnames['questions_out'], 'w')
    questions_id_file = open(fnames['questions_id_out'], 'w')
    questions_lengths_file = open(fnames['questions_lengths_out'], 'w')
    answers_file = open(fnames['answers_file_out'], 'w')
    coco_image_id = open(fnames['coco_image_id_out'], 'w')

    questions = json.load(open(question_fname, 'r'))
    ques = questions['questions']
    if args.split == 'train' or args.split == 'val':
        qa = json.load(open(annotation_fname, 'r'))
        qa = qa['annotations']

    # pbar = progressbar.ProgressBar()
    print('Dumping questions, answers, questionIDs, imageIDs, and questions lengths to text files...')
    with click.progressbar(list(zip(range(len(ques)), ques)), label='Processing...') as pbar:
        for (i, q) in pbar:
            questions_file.write((q['question'] + '\n'))
            questions_lengths_file.write(
                (str(len(nlp(q['question']))) + '\n'))
            questions_id_file.write((str(q['question_id']) + '\n'))
            coco_image_id.write((str(q['image_id']) + '\n'))
            if args.split == 'train' or args.split == 'val':
                if args.answers == 'modal':
                    answers_file.write(
                        get_modal_answer(qa[i]['answers']))
                elif args.answers == 'all':
                    answers_file.write(
                        get_all_answer(qa[i]['answers']))
                answers_file.write('\n')

    print('completed dumping', fnames['data_split'])
    print('Files:\n{0}'.format(pretty(fnames)))


if __name__ == "__main__":
    main()
