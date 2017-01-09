#!/usr/bin/env python3

# IMPORTANT: Do not rely on constants as experiment parameters. Keep track of
# all experiment params in a central dictionary, for easy pickling (this
# aids reproducibility a lot).

import logging

from visualqa.bow_vgg_baseline import BoWVGGBaseline


def main():
    # Dead skeleton code; do not use!
    logging.basicConfig(level=logging.INFO)
    baseline = BoWVGGBaseline()
    baseline.train()


if __name__ == '__main__':
    main()
