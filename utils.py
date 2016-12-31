"""
Source: https://github.com/avisingh599/visual-qa/
"""

import os

import operator
from itertools import zip_longest
from collections import defaultdict

def selectFrequentAnswers(questions_train, answers_train, images_train, maxAnswers):
    answer_fq= defaultdict(int)
    #build a dictionary of answers
    for answer in answers_train:
        answer_fq[answer] += 1

    sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:maxAnswers]
    top_answers, top_fq = zip(*sorted_fq)
    new_answers_train=[]
    new_questions_train=[]
    new_images_train=[]
    #only those answer which appear int he top 1K are used for training
    for answer,question,image in zip(answers_train, questions_train, images_train):
        if answer in top_answers:
            new_answers_train.append(answer)
            new_questions_train.append(question)
            new_images_train.append(image)

    return (new_questions_train,new_answers_train,new_images_train)


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

