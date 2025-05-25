# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import os, sys
sys.path.append(os.getcwd())

from sfm_learner.models.model_wrapper import ModelWrapper
from sfm_learner.models.model_checkpoint import ModelCheckpoint
from sfm_learner.trainers.horovod_trainer import HorovodTrainer
from sfm_learner.utils.config import parse_train_file
from sfm_learner.utils.load import set_debug, filter_args_create
from sfm_learner.utils.horovod import hvd_init, rank


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('--file', type=str, help='Input file (.yaml)')
    parser.add_argument('--ckpt', type=str, help='Input file (.ckpt)')
    args = parser.parse_args()
    assert args.file.endswith('.yaml'), \
        'You need to provide a .yaml file'
    if args.ckpt is not None:
        assert args.ckpt.endswith('.ckpt'), \
        'You need to provide a .ckpt file'
    return args


def train(file, ckpt):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file, ckpt)

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = None

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath == '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    train(args.file, args.ckpt)
