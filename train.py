#import importlib
#import os
import argparse
#import math
#import time
#import torch
#from omegaconf import OmegaConf

#import pytorch_lightning as pl

#from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.utils.extra import cprint, find_best_checkpoint

#from src.utils.train_utils import get_num_gaussians_from_checkpoint



def train(config):
    cprint("Training","blue")
    
    
 
 
def main(args):
    
    cprint("Starting Train.py","green")
    
    if args.method == "train":


        train()    
    
    
if __name__ == "__main__":
    # Get arguments from CMDLine
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument("--method", choices=["train","evaluate"])
    args = parser.parse_args()

    main(args)