import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import argparse
import utils_binary as utils
import Model_binary as Model
import sys
from sys import platform
import os
import time
#========================#
#    Global Parameter    #
#========================#
parser = argparse.ArgumentParser()

parser.add_argument('--Dataset'         , type = str, default = 'cifar10')
parser.add_argument('--Model_1st'       , type = str, default = 'ResNet')
parser.add_argument('--Model_2nd'       , type = str, default = '20_cifar10_2')
parser.add_argument('--BatchSize'       , type = int, default = 100)
parser.add_argument('--Epoch'           , type = int, default = 10)
parser.add_argument('--Debug_Mode'      , type = int, default = 0)

FLAGs = parser.parse_args()

Model_Name = FLAGs.Model_1st + '_' + FLAGs.Model_2nd
IS_HYPERPARAMETER_OPT = False

print('\n\033[1;32;40mMODEL NAME\033[0m :\033[1;37;40m {MODEL_NAME}\033[0m' .format(MODEL_NAME = Model_Name))
#============#
#    Path    #
#============#
# For Loading Dataset
Dataset_Path = '../dataset/' + FLAGs.Dataset

if FLAGs.Dataset=='ade20k':
    Dataset_Path = Dataset_Path + '/ADEChallengeData2016'
elif FLAGs.Dataset=='ILSVRC2012':
    Dataset_Path = Dataset_Path + '/imagenet-data'
Y_pre_Path   = FLAGs.Model_1st + '_Y_pre/' + FLAGs.Dataset

# For Saving Result Picture of Testing
train_target_path = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
valid_target_path = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 
test_target_path  = '../result/' + FLAGs.Dataset + '/' + FLAGs.Model_1st + '/' 

# For Saving Result in .npz file
train_Y_pre_path  = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
valid_Y_pre_path  = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'
test_Y_pre_path   = 'Y_pre/' + FLAGs.Model_1st + '/' + FLAGs.Dataset + '/'

diversify_layers = []

#==============#
#    Define    #
#==============#
def main(argv):
    # For Loading Trained Model
    # -- ResNet --
    if FLAGs.Model_1st == 'ResNet':
        ## ResNet-110
        if FLAGs.Model_2nd == '110_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_110_cifar10_0_layer1_to_layer54_mode1_Filter_Similar95_93_layer55_to_layer108_mode1_Filter_Similar95_94_layer109_to_layer162_mode1_Filter_Similar95_94/'
            Model = '50.ckpt'
        ## ResNet-56
        if FLAGs.Model_2nd == '56_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_56_cifar10_0_layer1_to_layer27_mode1_Filter_Similar30_29_layer28_to_layer54_mode1_Filter_Similar30_29_layer55_to_layer81_mode1_Filter_Similar30_29/'
            Model = '50.ckpt'
        if FLAGs.Model_2nd == '56_Binary':
            Model_Path = 'Model/ResNet_Model/ResNet_56_Binary_98_cifar10_2018.04.25_Filter_Angle10_40/' 
            Model = 'best.ckpt'
        ## ResNet-32
        if FLAGs.Model_2nd == '32_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_32_cifar10_0_99_2018.02.09_Filter_Magnitude31_47/'
            Model = 'best.ckpt'
        ## ResNet-20
        if FLAGs.Model_2nd == '20_cifar10_2':
            Model_Path = 'Model/ResNet_Model/ResNet_20_cifar10_2_layer0_to_layer9_mode1_Filter_Similar30_29_layer10_to_layer18_mode1_Filter_Similar30_30_layer19_to_layer27_mode1_Filter_Similar30_30/'
            Model = '50.ckpt'
            diversify_layers = ['layer14', 'layer16', 'layer17']
        if FLAGs.Model_2nd == '20_Ternary':
            Model_Path = 'Model/ResNet_Model/ResNet_20_Ternary_98_cifar10_2018.04.14/'
            Model = '400.ckpt'
        if FLAGs.Model_2nd == '20_Binary':
            Model_Path = 'Model/ResNet_Model/ResNet_20_Binary_best/'
            Model = 'best.ckpt'
        ## ResNet-50
        if FLAGs.Model_2nd == '50':
            Model_Path = 'Model/ResNet_Model/ResNet_50_74_cifar10_2018.03.27/'
            Model = '1.ckpt'
    # -- DenseNet --
    if FLAGs.Model_1st == 'DenseNet':
        ## DenseNet_40_12
        if FLAGs.Model_2nd == '40_12':
            Model_Path = 'Model/DenseNet_Model/DenseNet_40_12_99_cifar10_2018.03.06/'
            Model = '300.ckpt'
    # -- MobileNet --
    if FLAGs.Model_1st == 'MobileNet':
        ## MobileNet_100_100
        if FLAGs.Model_2nd == '100_100':
            Model_Path = 'Model/MobileNet_Model/MobileNet_100_100_65_cifar10_2018.04.07/'
            Model = '1.ckpt'
    # -- BinaryConnect --
    if FLAGs.Model_1st == 'BinaryConnect':
        ## BinaryConnect_cifar10
        if FLAGs.Model_2nd == 'cifar10':
            Model_Path = 'Model/BinaryConnect_Model/BinaryConnect_cifar10_31_cifar10_2018.04.18/'
            Model = '2.ckpt'
    # -- Testing --
    Model_Path, Model = utils.run_debuging( 
        FLAGs               = FLAGs         ,
        Dataset_Path        = Dataset_Path  ,
        debug_model_path    = Model_Path    ,
        debug_model         = Model         )
    
if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)

