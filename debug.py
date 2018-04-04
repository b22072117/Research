import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import argparse
import csv

import utils
import Model

import sys
from sys import platform
import os
import time


# Check the untrainable variable is same or not 
Dataset = "cifar10"
Model_Path_1 = "Model/ResNet_Model/ResNet_20_cifar10_2_99_cifar10_2018.02.06_Filter_Angle10_68_Rebuild54/"
Model_Path_2 = "Model/ResNet_Model/ResNet_20_cifar10_2_99_cifar10_2018.02.06_Filter_Angle10_68_Rebuild54_43/"
Model_1 = "160.ckpt"
Model_2 = "160.ckpt"

utils.check_untrainable_variable_no_change(
    Dataset      = Dataset      ,
    Model_Path_1 = Model_Path_1 ,
    Model_Path_2 = Model_Path_2 ,
    Model_1      = Model_1      ,
    Model_2      = Model_2      )