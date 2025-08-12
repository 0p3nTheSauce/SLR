import argparse
import torch 
import os

from utils import enum_dir
from torchvision.transforms import v2
# from  torchvision.models.video.resnet import  R3D_18_Weights , r3d_18
# from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# import models.pytorch_r3d as resnet_3d


#local imports
from video_dataset import VideoDataset
from configs import load_config, print_config
import numpy as np
import random
import wandb
from models.pytorch_r3d.py import Resnet3D18_basic
from train import train_loop

def train(wandb_run, load=None, weights=None, save_every=5, recover=False):
    