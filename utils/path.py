import os
import glob
import re
import yaml
from os.path import exists, dirname, abspath, join
from pathlib import Path
from configparser import ConfigParser

config = ConfigParser()
config.read(join(dirname(dirname(abspath(__file__))),"airsim_gym", 'config.ini'))


def get_export_path(args):
    """
    Return export path to save tf/keras model
    """
    save_dir = increment_path(get_model_name(args))
    create_path_if_does_not_exit(save_dir)
    save_run_settings(save_dir, args)
    return save_dir


def get_model_name(args):
    if hasattr(args, "algorithm"):
        return 'results/{}/{}/{}'.format(
            args.experiment_name,
            args.algorithm,
            args.model_type,
        )
    else:
        return 'results/{}/{}'.format(
            args.experiment_name,
            args.model_type,
        )


def increment_path(path, sep=''):
    # Increment path, i.e. logs/ppo --> logs/ppo{sep}0, logs/ppo{sep}1 etc.
    path = Path(path)  # os-agnostic
    if not path.exists():
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def create_path_if_does_not_exit(log_dir):
    """
    Create save dirs
    """
    if not exists(log_dir):
        os.makedirs(log_dir)
    checkpoints_dir = Path("{}/checkpoints/".format(log_dir))
    tensorboard_dir = Path("{}/tensorboard/".format(log_dir))
    best_model_dir = Path("{}/best_model".format(log_dir))
    if not exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not exists(best_model_dir):
        os.makedirs(best_model_dir)


def save_run_settings(save_dir, args):
    # Save run settings
    save_dir = Path(save_dir)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(dict(config._sections['airsim_settings']), f, sort_keys=False)
        yaml.dump(dict(config._sections['car_agent']), f, sort_keys=False)
