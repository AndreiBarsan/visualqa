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

# Hint: set your appropriate user and host for Euler in your '~/.ssh/config'!
# Hint: To pass multiple arguments, to a fabric command, use:
#  $ fab euler:run,some-label,foo,bar
# Hint: configure this host to point to a GPU-enabled machine on AWS running
# Andrei's prebuilt image, with AMI ID [ami-034aee63].

Examples:
    `fab preprocess` To perform preprocessing on the remote instance.
    `fab train` To run the tool on an already configured AWS instance.
"""

from os.path import join as pjoin

from fabric.api import *
from fabric.contrib.project import rsync_project as rsync

env.use_ssh_config = True



@hosts('aws-gpu')
def preprocess() -> None:
    _sync_code()
    work_dir = "/home/ubuntu/vqa/visualqa"
    with cd(work_dir):
        # Ensure we have the assets for tokenizing stuff.
        run(_as_conda('python -m spacy.en.download || echo Spacy OK'))
        # Extract the useful parts of the JSON inputs as text files.
        # TODO(andrei): Parameterize this properly.
        split = 'val'
        answers = 'all'
        run(_as_conda('./preprocess.py -dataroot /data/vqa -split {0} '
                      '-answers {1}'.format(split, answers)))


@hosts('aws-gpu')
def train(run_label: str='aws-exp', in_screen: str='True') -> None:
    """Runs the TF pipeline on commodity hardware with no job queueing."""
    in_screen = in_screen.lower() in ['1', 'true']
    print("Running AWS task with label [{0}], {1}in a screen.".format(
        run_label, "" if in_screen else "NOT "
    ))
    work_dir = "/home/ubuntu/vqa/experiments"
    print("Using work dir {0}.".format(work_dir))
    _sync_code()

    with cd(work_dir):
        ts = '$(date +%Y%m%dT%H%M%S)'
        tf_command = ('t=' + ts + ' && mkdir $t && cd $t && python ' + _run_experiment(run_label))

        if in_screen:
            _in_screen(_as_conda(tf_command), 'vqa_experiment_screen',
                       shell_escape=False, shell=False)
        else:
            run(_as_conda(tf_command), shell=False, shell_escape=False)


@hosts('aws-gpu')
def eval(experiment_id: str) -> None:
    """Evaluates the accuracy of the model trained by the given experiment."""
    _sync_code()
    print("WARNING: ignoring specified experiment ID {0}".format(experiment_id))

    root = pjoin('/data', 'vqa', 'models')
    # TODO(andrei): Always put these in the folder with experiment ID.
    model_fname = 'mlp_num_hidden_units_1024_num_hidden_layers_3.json'
    weight_fname = 'mlp_num_hidden_units_1024_num_hidden_layers_3_epoch_99.hdf5'
    results_fpath = pjoin('/tmp/', 'results-changeme.txt')

    model_fpath = pjoin(root, model_fname)
    weight_fpath = pjoin(root, weight_fname)

    with cd('/home/ubuntu/vqa/visualqa'):
        run(_as_conda(
            'python evaluateMLP.py -model {0} -weights {1} -results {2} '
            '-dataroot /data/vqa'.format(model_fpath, weight_fpath,
                                         results_fpath)))


def _run_experiment(run_label: str) -> str:
    """This is the command for training the model.

    It is called inside a screen right away when running on AWS, and submitted
    to LFS using 'bsub' on Euler.
    """
    # return "../../visualqa/main.py"
    return '../../visualqa/trainMLP.py -dataroot /data/vqa -batch_size 1024'


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

