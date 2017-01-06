"""Fabric deployment file for remote model training.

Make sure that 'env.hosts' points to wherever you want to train your model, and
that the remote host has tensorflow installed.

# Hint: set your appropriate user and host for Euler in your '~/.ssh/config'!
# Hint: To pass multiple arguments, to a fabric command, use:
#  $ fab euler:run,some-label,foo,bar
# Hint: configure this host to point to a GPU-enabled machine on AWS running
# Andrei's prebuilt image, with AMI ID [ami-034aee63].

Examples:
    `fab -l`            Lists all available commands.
    `fab preprocess`    To perform preprocessing on the remote instance.
    `fab train`         To run the tool on an already configured AWS instance.

Notes:
    Uses a Python 3 fork of Fabric (http://www.fabfile.org/).
    Please install 'Fabric3' to use this, NOT the vanilla 'fabric'.

```bash
    pip install Fabric3
```

Note that even on Anaconda, you still have to use 'pip' to install 'Fabric3'.
"""

import io
from os.path import join as pjoin

from fabric.api import *
from fabric.contrib.project import rsync_project as rsync

from visualqa.utils import args_to_flags

# Name of Anaconda environment used to run the Python 2 VQA evaluation code.
PYTHON2_ENV_NAME = 'dl-2.7'
env.use_ssh_config = True


@task
@hosts('aws-gpu')
def preprocess(*args, **kw) -> None:
    """Extracts useful stuff from the JSON data files.

    Examples
        Running `fab preprocess:<args>` forwards the args to the
        `preprocess.py` script running remotely. `fab preprocess:--help`
        shows all available flags, and so does `./preprocess.py --help`.
    """
    _sync_code()
    work_dir = "/home/ubuntu/vqa/visualqa"
    with cd(work_dir):
        # Ensure we have the assets for tokenizing stuff.
        run(_as_conda('python -m spacy.en.download || echo Spacy OK'))

        # Extract the useful parts of the JSON inputs as text files.
        run(_as_conda('./visualqa/preprocess.py -dataroot /data/vqa {0} '.format(
            args_to_flags(args, kw))))


@task
@hosts('aws-gpu')
def train(run_label: str='aws-exp', in_screen: str='True', *args, **kw) -> None:
    """Runs the TF pipeline on commodity hardware with no job queueing.

    Examples
        `fab train:baseline-exp,false,-h` Starts an experiment,
        with 'baseline-exp' as its label, NOT running inside a screen,
        and with the flag '-h' to be passed to the training script. This does
        not train anything, and only shows the help message.
    """
    in_screen = parse_bool(in_screen)
    print("Running AWS task with label [{0}], {1}in a screen.".format(
        run_label, "" if in_screen else "NOT "
    ))
    work_dir = "/home/ubuntu/vqa/experiments"
    print("Using work dir {0}.".format(work_dir))
    _sync_code()

    with cd(work_dir):
        ts = '$(date +%Y%m%dT%H%M%S)' + '-' + run_label
        tf_command = ('t=' + ts + ' && mkdir $t && cd $t && python ' +
                      _run_experiment(*args, **kw))

        if in_screen:
            _in_screen(_as_conda(tf_command), 'vqa_experiment_screen',
                       shell_escape=False, shell=False)
        else:
            run(_as_conda(tf_command), shell=False, shell_escape=False)


@task
@hosts('aws-gpu')
def setup_conda() -> None:
    # Creates a Python 2.7 environment required for the official VQA
    # evaluation suite.
    run('conda create -y --quiet --name dl-2.7 python=2.7 || echo Environment '
        '\'dl-2.7\' probably already exists.')
    run('source activate dl-2.7 && '
        'conda install -y --quiet scikit-learn scikit-image matplotlib spacy')
    run('. activate ml && conda install spacy')
    run('. activate ml && conda install keras')
    # TODO-LOW(andrei): Code for setting up the main environment as well.
    # Note: this env is already included in Andrei's AWS AMI (if you're using
    # that, and are not on Azure) under the name 'ml'.
    # run('conda create -y --name ml python=3.5')


@task(aliases=['ls', 'lsexp', 'ls_exp'])
@hosts('aws-gpu')
def list_experiments() -> None:
    """Lists all available experiment names.

    The experiment ID is its folder name. This is what should be passed to
    `eval` in order to perform the evaluation of that run.
    """
    result = run('ls -lt /home/ubuntu/vqa/experiments/', stdout=io.StringIO())
    print("Available experiment results (newest first):")
    for line in result.splitlines():
        if 'total' in line: continue

        # Sample line: drwxrwxr-x 2 ubuntu ubuntu 4096 Jan  3 18:51 20170101T141639
        perms, link_count, user, group, size, month, day, time, name = line.split()
        print("\tID {0} ({1} {2} at {3})".format(name, month, day, time))


@task
@hosts('aws-gpu')
def eval(experiment_id: str, epoch: str='-1', *args, **kw) -> None:
    """Evaluates the accuracy of the model trained by the given experiment.

    TODO(andrei): Have 'train' script print out a copy-paste-friendly version
    of the corresponding eval script.

    As of Jan 3rd, this takes ~2:57 total.
    """
    _sync_code()
    epoch = int(epoch)
    eval_on = 'val'
    experiment_label = 'baseline' # TODO(andrei): Set dynamically.
    dataset_label_map = {
        'val': 'val2014',
        'train': 'train2014',
        'test-dev': 'test-dev2015',
        'test': 'test2015'
    }
    # Whether to assume the actual answers have already been computed. Useful
    # if you only want to quickly recompute the accuracy and other stats.
    skip_answer_computation = parse_bool(
        kw.get('skip_answer_computation', 'False'))
    if 'skip_answer_computation' in kw: del kw['skip_answer_computation']
    if skip_answer_computation:
        print("WARNING: will NOT recompute the answers from scratch, and will "
              "just recompute statistics. Ignoring epoch parameter.")

    # root = pjoin('/data', 'vqa', 'models')
    root = pjoin('/home', 'ubuntu', 'vqa', 'experiments')
    experiment_folder = pjoin(root, experiment_id)

    model_fname = 'model.json'
    weight_fnames_raw = run('ls {0}/*.hdf5'.format(experiment_folder),
                            stdout=io.StringIO())
    weight_fnames = weight_fnames_raw.splitlines()
    epoch_weight_fnames = [(_get_epoch_number(weight_fname), weight_fname)
                           for weight_fname in weight_fnames]
    epoch_weight_fnames.sort(key=lambda tup: tup[0])

    if epoch == -1:
        # Get the latest one
        weight_fpath = epoch_weight_fnames[-1][1]
    else:
        try:
            weight_fpath = next(fn for fn_epoch, fn in epoch_weight_fnames
                                if fn_epoch == epoch)
        except StopIteration:
            # No weights saved at that epoch. Give the user info on what the
            # closest epoch we DO have a dump for.
            deltas = [(abs(fn_epoch - epoch), fn_epoch)
                      for fn_epoch, _ in epoch_weight_fnames]
            closest = min(deltas, key=lambda tup: tup[0])[1]
            print("Could not find checkpoint for epoch #{0}. Closest epoch "
                  "with available weights is #{1}.".format(epoch, closest))
            return

    print("Will use weights from file: [{0}]".format(weight_fpath))

    results_fpath = pjoin(experiment_folder, 'results-{0}.txt'.format(eval_on))

    # TODO(andrei): Fully support evaluating on TRAINING data as well (and
    # test that it works).
    results_json_fpath = pjoin(experiment_folder,
                               'OpenEnded_mscoco_{0}_{1}_results.json'.format(
                                   dataset_label_map[eval_on],
                                   experiment_label
                               ))

    model_fpath = pjoin(experiment_folder, model_fname)
    with cd('/home/ubuntu/vqa/visualqa'):
        if not skip_answer_computation:
            # Generate the predictions on the validation set...
            run(_as_conda(
                'python ./visualqa/evaluateMLP.py -model {0} -weights {1} -results {2} '
                '-results_json {3} -dataroot /data/vqa'.format(
                    model_fpath, weight_fpath, results_fpath, results_json_fpath)))

        # ...and measure all sorts of cool stats.
        with cd('VQA'):
            VQA_eval_command = 'python PythonEvaluationTools/vqaEvalDemo.py ' \
                               '{0} {1}'.format(experiment_folder,
                                                args_to_flags(args, kw))
            run(_as_conda(VQA_eval_command, PYTHON2_ENV_NAME))


def _run_experiment(*args, **kw) -> str:
    """This is the command for training the model.

    It is called inside a screen right away when running on AWS, and submitted
    to LFS using 'bsub' on Euler.
    """
    # return "../../visualqa/main.py"
    if '-batch_size' not in kw:
        kw['-batch_size'] = 512
    if '-dataroot' not in kw:
        kw['-dataroot'] = '/data/vqa'

    return '../../visualqa/visualqa/trainMLP.py {0}'.format(args_to_flags(args, kw))


def _sync_code(remote_code_dir='/home/ubuntu/vqa/visualqa') -> None:
    """Copies the code to the remote host. Does NOT copy data.

    This is simply because the VQA dataset on which we are operating is over
    30 GiB, and it would be infeasible to try to sync it every time.
    """
    run('mkdir -p {0}'.format(remote_code_dir))
    rsync(remote_dir=remote_code_dir,
          local_dir='.',
          exclude=['.git', '.idea', '*__pycache__*', '*.pyc'])


def _as_conda(cmd: str, env_name='ml') -> str:
    """Ensures the command gets run in the specified anaconda env."""

    # Sourcing the bashrc to ensure LD_LIBRARY_PATH is sane.
    return "source /home/ubuntu/.bashrc && " \
           "source /home/ubuntu/bin/anaconda3/bin/activate {0} && " \
           "{1}".format(env_name, cmd)


def _in_screen(cmd: str, screen_name: str, **kw) -> None:
    """Runs the specified command remotely inside a persistent screen.

    This allows the command to properly return after kicking off a job.
    The screen persists into a regular 'bash' after the command completes.

    Notes
        One can use `screen -ls` and `screen -r <id>` to list active screens,
        and to reattach to a specific screen on the host running the jobs.
    """
    screen = "screen -dmS {0} bash -c '{1} ; exec bash'".format(screen_name, cmd)
    print("Screen to run: [{0}]".format(screen))
    run(screen, pty=False, **kw)


def _get_epoch_number(weight_fname: str) -> int:
    """Extracts epoch number from a hdf5 weight dump file name."""

    nr_str = weight_fname[weight_fname.rfind('_') + 1:
                          weight_fname.rfind('.')]
    return int(nr_str)


def parse_bool(input: str) -> bool:
    """Dirty, dirty, parsing."""
    return input.lower() in ['1', 'true']

