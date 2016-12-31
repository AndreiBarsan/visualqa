"""Simple baseline using word vector adding an VGG image embeddings."""

import logging

log = logging.getLogger('bow_vgg_baseline')


class BoWVGGBaseline(object):

    def __init__(self, **kw):
        pass

    def train(self):
        log.info("Training BoW VGG Baseline...")
