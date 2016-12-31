# visualqa
Visual Question Answering project for ETHZ's 2016/17 Deep Learning class.


## Layout

This project is designed to be run on AWS/Azure GPU instances.
The format I'm currently using expects the following folder structure for the
data on the remote instance:

```
/data
└── vqa
    ├── Annotations
    ├── Images
    │   └── mscoco
    │       ├── test2015
    │       ├── train2014
    │       └── val2014
    ├── Preprocessed
    └── Questions
```

This folder structure uses the format expected by the official VQA evaluation
framework from https://github.com/VT-vision-lab/VQA.


## Setup

 0. Provision an AWS/Azure GPU instance with Ubuntu.
 1. Prepare the data, as indicated in the 'Layout' section.
 2. Run `fab aws:run` or similar to train the model.
   * TODO(andrei): Actually implement this + auto evaluation, etc.
