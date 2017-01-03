"""
Source: https://github.com/avisingh599/visual-qa/
"""

import os

import operator
from itertools import zip_longest
from collections import defaultdict
from typing import List, Dict


def select_frequent_answers(questions_train, answers_train, images_train, maxAnswers):
    answer_fq= defaultdict(int)
    #build a dictionary of answers
    for answer in answers_train:
        answer_fq[answer] += 1

    sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:maxAnswers]
    top_answers, top_fq = zip(*sorted_fq)
    new_answers_train=[]
    new_questions_train=[]
    new_images_train=[]

    for answer,question,image in zip(answers_train, questions_train, images_train):
        if answer in top_answers:
            new_answers_train.append(answer)
            new_questions_train.append(question)
            new_images_train.append(image)

    return (new_questions_train, new_answers_train, new_images_train)


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def mkdirp(path):
    """Emulates 'mkdir -p' functionality."""
    try:
        os.makedirs(path)
    except FileExistsError:
        # The folder already exists. We're good.
        pass


def lines(fpath: str) -> List[str]:
    with open(fpath, 'r') as file:
        return file.read().splitlines()


def args_to_flags(args: List, kw_map: Dict) -> str:
    return " ".join(args) + kw_to_flags(kw_map)


def kw_to_flags(kw_map: Dict) -> str:
    """Converts a list of keyword arguments to a string.

    Assumes keys already have '--' prefixes.
    """
    return " ".join(("{0}={1}".format(k, v) for k, v in kw_map.items()))




