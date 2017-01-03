# visualqa
Visual Question Answering project for ETHZ's 2016/17 Deep Learning class.

## Protips

`TODO(name):` Signals the author of the TODO, who will probably know a lot about
it, but it's not necessarily a commitment---anyone can tackle any TODO. It's
actually encouraged!


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

 1. Provision an AWS/Azure GPU instance with Ubuntu.
 1. Prepare the data, as indicated in the 'Layout' section. Run
    `fab setup_conda` to install some dependencies of the evaluation code.
 1. Run `fab preprocess` to preprocess the data for training. Run
    `preprocess.py -h` for more information and available options.
 1. Run `fab train` to train the default model (i.e., the baseline at the
    moment). This will kick off in a screen on the host, and the script will
    detach. You can SSH into the remote host and attach to that script to
    check the progress!
 1. After the training completes, run `fab eval` to generate the predictions
    on the validation set, and crunch them through the VQA evaluation tools,
    producing many cool stats, such as an accuracy breakdown based on
    question and answer types.
    This task is a little finniky and may need some manual tweaking in
    `fabfile.py` at the moment in order to work. Also, bear in mind that the
    `VQA` subproject requires Python 2.7.
    # TODO(andrei): Update this info once tooling improves.
