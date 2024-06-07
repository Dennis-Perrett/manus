import importlib
import os

import math
import time
import torch
from omegaconf import OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from src.utils.extra import cprint, find_best_checkpoint
from src.utils.train_utils import get_num_gaussians_from_checkpoint

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

os.environ['SLURM_JOB_NAME'] = 'bash'
torch.set_float32_matmul_precision('medium')  # 'high'


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



@hydra.main(config_path="./config", config_name="OBJ_GAUSSIAN", version_base='1.1')
def main(config):
    cprint(OmegaConf.to_yaml(config),'red')
    cprint(config.checkpoint, 'green')
    cur_path = os.getcwd()

    if config.trainer.mode == 'train':
        if len(os.listdir(cur_path)) != 0:
            os.system("rm -r *")
            cprint("------------------------------------------------------", 'red')
            cprint(f"Cleaning up the directory.. {cur_path}", 'red')
            cprint("------------------------------------------------------", 'red')

    if config.trainer.mode == 'train':
        cprint("Starting Training... Setting up OmegaConfig", 'green')
        save_path = os.path.join(cur_path, "config.yaml")
        if os.path.exists(save_path):
            time_now = math.ceil(time.time())
            save_path = os.path.join(cur_path, f"config_{time_now}.yaml")
        cprint(f"Saving config to {save_path}", 'blue')
        OmegaConf.save(config, save_path)

    ckpt_dir = os.path.join(cur_path, "checkpoints/")
    
    cprint("Checking if checkpoint exists","green")
    cprint(f"Config Checkpoint: {config.checkpoint}", 'magenta')
    if config.checkpoint:
        cprint(f"Checkpoint Exists {config.checkpoint}", 'magenta')
        if config.checkpoint == "best":
            config.checkpoint = find_best_checkpoint(ckpt_dir)
        else:
            config.checkpoint = os.path.join(ckpt_dir, config.checkpoint)
        cprint(f"Loading from the checkpoint: {config.checkpoint}", 'green')
    else:
        cprint(f"Creating checkpoint dirs {ckpt_dir}","blue")
        os.makedirs(ckpt_dir, exist_ok=True)
        config.checkpoint = None
        
    pl.seed_everything(config.trainer.seed),

    cprint("Calling train function", 'green')
    train(config, mode=config.trainer.mode, ckpt_dir=ckpt_dir)


def train(config, mode, ckpt_dir):
    cprint("Running Train","green")
    cprint(f"Train Mode: {mode}",'magenta')
    if mode != 'test':
        cprint("Mode is NOT test",'red')
        callbacks = [
            ModelCheckpoint(
                # monitor="loss",
                dirpath=ckpt_dir,
                filename="{epoch:03d}-{step}-{loss:.6f}",
                save_top_k=-1,
                mode="min",
                every_n_epochs=1,
                verbose=True
            ),
            LearningRateMonitor(logging_interval='step'),
        ]

        loggers = []
        if mode != 'test':
            if 'csv' in config.trainer.loggers:
                loggers.extend([CSVLogger('.', name='logs', version='csv_logs')])
    else:
        callbacks = []
        loggers = []

    if mode == 'debug':
        trainer = pl.Trainer(fast_dev_run=True)
    else:
        cprint("Running pure train mode without debug or test","blue")
        ## DDP Fails for multiple models and optimizers
        # strategy = 'ddp'
        ## DDP_parameter_false works for multiple models and optimizers but slow
        # strategy = 'ddp_find_unused_parameters_false'
        trainer = pl.Trainer(
            devices=config.trainer.gpus,
            accelerator='gpu',
            callbacks=callbacks,
            logger=loggers,
            **config.trainer.pl_vars
        )

    cprint(f"Config Checkpoint is: {config.checkpoint}", "magenta")
    if config.checkpoint is not None:
        num_gaussians = get_num_gaussians_from_checkpoint(config.checkpoint)
        config.model.opts.num_gaussians = num_gaussians
        module = hydra.utils.instantiate(config.module, _recursive_=False)
        module = module.load_from_checkpoint(checkpoint_path=config.checkpoint, opts=config.opts,
                                             mode=config.trainer.mode)
    else:
        module = hydra.utils.instantiate(config.module, _recursive_=False)

    if config.trainer.torch_compile_mode is not None:
        module = torch.compile(module, mode=config.trainer.torch_compile_mode)

    if mode == 'test':
        trainer.test(module)
    else:
        trainer.fit(module)


if __name__ == "__main__":
    main()
