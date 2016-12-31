"""Fabric deployment file for remote model training.

TODO(andrei): fabric is kind of deprecated. Use 'pyinvoke'.

Uses a Python 3 fork of Fabric (http://www.fabfile.org/).
Please install 'Fabric3' to use this, NOT the vanilla 'fabric'.

```bash
    pip install Fabric3
```

Note that even on Anaconda, you still have to use 'pip' to install 'Fabric3'.

Make sure that 'env.hosts' points to wherever you want to train your model, and
that the remote host has tensorflow installed.

Examples:
    `fab aws:run` To run the tool on an AWS instance you already configured.
"""


from __future__ import with_statement

import os

from fabric.api import *
from fabric.contrib.project import rsync_project as rsync

env.use_ssh_config = True


# Hint: set your appropriate user and host for Euler in your '~/.ssh/config'!
# Hint: To pass multiple arguments, to a fabric command, use:
#  $ fab euler:run,some-label,foo,bar
# Hint: configure this host to point to a GPU-enabled machine on AWS running
# Andrei's prebuilt image, with AMI ID [ami-034aee63].
@hosts('aws-gpu')
def aws(sub='run', label='aws-tesla'):
    if sub == 'run':
        _run_commodity(label)
    elif sub == 'preprocess':
        _preprocess_commodity()
    else:
        raise ValueError("Not yet supported.")


# TODO(andrei): Do this properly, e.g. supporting subcommands, if necessary.
# def local(sub='run', label='local'):
#     flocal('python learn.py')


def _preprocess_commodity() -> None:
    _sync_code()
    work_dir = "/home/ubuntu/vqa/visualqa"
    with cd(work_dir):
        # Ensure we have the assets for tokenizing stuff.
        run(_as_conda('python -m spacy.en.download'))
        # Extract the useful parts of the JSON inputs as text files.
        run(_as_conda('./preprocess.py -dataroot /data/vqa'))


def _run_commodity(run_label: str) -> None:
    """Runs the TF pipeline on commodity hardware with no job queueing."""
    work_dir = "/home/ubuntu/vqa/experiments"
    print("Using work dir {0}.".format(work_dir))
    _sync_code()

    with cd(work_dir):
        ts = '$(date +%Y%m%dT%H%M%S)'
        tf_command = ('t=' + ts + ' && mkdir $t && cd $t && python ' + _run_experiment(run_label))

        # TODO(andrei): Flag to disable screen.
        run(_as_conda(tf_command), shell=False, shell_escape=False)
        # _in_screen(_as_conda(tf_command), 'vqa_experiment_screen',
        #            shell_escape=False, shell=False)


def _eval_commodity(experiment_id: str) -> None:
    """Evaluates the accuracy of the model trained by the given experiment."""
    raise Exception("Not yet implemented.")


def _run_experiment(run_label: str) -> str:
    """This is the command for training the model.

    It is called inside a screen right away when running on AWS, and submitted
    to LFS using 'bsub' on Euler.
    """
    # return "../../visualqa/main.py"
    return '../../visualqa/trainMLP.py'


def _sync_code(remote_code_dir='/home/ubuntu/vqa/visualqa') -> None:
    """Copies the code to the remote host. Does NOT copy data.

    This is simply because the VQA dataset on which we are operating is over
    30 GiB, and it would be infeasible to try to sync it every time.
    """
    run('mkdir -p {0}'.format(remote_code_dir))
    rsync(remote_dir=remote_code_dir,
          local_dir='.',
          exclude=['.git', '.idea', '*__pycache__*'])


def _as_conda(cmd: str, env_name='ml') -> str:
    """Ensures the command gets run in the specified anaconda env."""

    # Sourcing the bashrc to ensure LD_LIBRARY_PATH is sane.
    return "source /home/ubuntu/.bashrc && " \
           "source /home/ubuntu/bin/anaconda3/bin/activate {0} && " \
           "{1}".format(env_name, cmd)


def _in_screen(cmd: str, screen_name: str, **kw) -> None:
    """Runs the specified command inside a persistent screen.

    This allows the command to properly return after kicking off a job.
    The screen persists into a regular 'bash' after the command completes.

    Notes
        One can use `screen -ls` and `screen -r <id>` to list active screens,
        and to reattach to a specific screen on the host running the jobs.
    """
    screen = "screen -dmS {0} bash -c '{1} ; exec bash'".format(screen_name, cmd)
    print("Screen to run: [{0}]".format(screen))
    run(screen, pty=False, **kw)

