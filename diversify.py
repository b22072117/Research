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
import copy
#========================#
#    Global Parameter    #
#========================#
parser = argparse.ArgumentParser()

parser.add_argument('--Dataset'        , type = str, default = 'cifar10')
parser.add_argument('--Model_1st'      , type = str, default = 'ResNet')
parser.add_argument('--Model_2nd'      , type = str, default = '20_cifar10_2')
parser.add_argument('--BatchSize'      , type = int, default = 128)
parser.add_argument('--Epoch'          , type = int, default = 160)
parser.add_argument('--epochs_per_eval', type = int, default = 1)
parser.add_argument('--Pruning_Strategy', type = str, default = 'Filter_Similar')
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


#==============#
#    Define    #
#==============#
def main(argv):
    Global_Epoch = 0
    Model_Path = None
    Model = None
    # -- Training --
    # For Loading Trained Weights
    if FLAGs.Model_1st == 'ResNet': 
        ## ResNet-110
        if FLAGs.Model_2nd == '110_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_110_cifar10_0_99_2018.02.08/'
            Model = '10.ckpt'
            Model_Path_s = 'Model/ResNet_Model/ResNet_110_cifar10_0_layer1_to_layer54_mode1_Filter_Similar50_49_layer55_to_layer108_mode1_Filter_Similar30_29/'
            Model_s = '50.ckpt'
            
            Model_Path_s = Model_Path
            Model_s = Model
            
            diversify_layers = {}
            diversify_layers.update({
                '0': {'layer': ['layer1'  , 'layer54' ], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.7, 'times': 1},
                '1': {'layer': ['layer55' , 'layer108'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.7, 'times': 1},
                '2': {'layer': ['layer109', 'layer162'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.7, 'times': 1},
            })
            pruned_model_path = None
            pruned_model = None
        ## ResNet-56
        if FLAGs.Model_2nd == '56_cifar10_0':
            Model_Path = 'Model/ResNet_Model/ResNet_56_cifar10_0_99_2018.02.09/'
            Model = '10.ckpt'
            
            Model_Path_s = 'Model/ResNet_Model/ResNet_56_cifar10_0_layer1_to_layer27_mode1_Filter_Similar50_49/'
            Model_s = '50.ckpt'
            
            #Model_Path_s = Model_Path
            #Model_s = Model
            
            diversify_layers = {}
            diversify_layers.update({
                '0': {'layer': ['layer1' , 'layer27'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.6, 'times': 1},
                '1': {'layer': ['layer28', 'layer54'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.6, 'times': 1},
                '2': {'layer': ['layer55', 'layer81'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.6, 'times': 1},
            })
            pruned_model_path = None
            pruned_model = None
        ## ResNet-20
        if FLAGs.Model_2nd == '20_cifar10_2':
            Model_Path = 'Model/ResNet_Model/ResNet_20_cifar10_2_99_2018.02.06/'
            Model = '10.ckpt'
            Model_Path_s = 'Model/ResNet_Model/ResNet_20_cifar10_2_layer1_to_layer9_mode1_Filter_Similar20_19_layer10_to_layer18_mode1_Filter_Similar20_19/'
            Model_s = '50.ckpt'
            
            Model_Path_s = Model_Path
            Model_s = Model
            
            diversify_layers = {}
            diversify_layers.update({
                '0': {'layer': ['layer1' , 'layer9' ], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.4, 'times': 1},
                '1': {'layer': ['layer10', 'layer18'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.4, 'times': 1},
                '2': {'layer': ['layer19', 'layer27'], 'type': 'diversify', 'mode' : '1', 'prune_propotion': 0.4, 'times': 1},
                #'0': {'layer': ['layer1', 'layer1'], 'type': 'diversify', 'mode' : '0', 'prune_propotion': 0.4, 'times': 1},
                #'0': {'layer': ['layer2', 'layer2'], 'type': 'diversify', 'mode' : '0', 'prune_propotion': 0.4, 'times': 1},
                #'2': {'layer': ['layer4', 'layer4'], 'type': 'diversify', 'mode' : '0', 'prune_propotion': 0.4, 'times': 1},
                #'3': {'layer': ['layer5', 'layer5'], 'type': 'diversify', 'mode' : '0', 'prune_propotion': 0.4, 'times': 1},
                #'4': {'layer': ['layer7', 'layer7'], 'type': 'diversify', 'mode' : '0', 'prune_propotion': 0.4, 'times': 1},
                #'5': {'layer': ['layer8', 'layer8'], 'type': 'diversify', 'mode' : '0', 'prune_propotion': 0.4, 'times': 1},
                
                #'1': {'layer': ['layer10', 'layer18'], 'type': 'diversify', 'mode' : '4', 'prune_propotion': 0.1, 'times': 6},
                #'2': {'layer': ['layer19', 'layer27'], 'type': 'diversify', 'mode' : '4', 'prune_propotion': 0.1, 'times': 6},
                #'1': {'layer': ['layer1' , 'layer9' ], 'type': 'diversify', 'mode' : '7', 'prune_propotion': None, 'times': 6},
                #'0': {'layer': ['layer10', 'layer18'], 'type': 'diversify', 'mode' : '7', 'prune_propotion': None, 'times': 6},
                #'5': {'layer': ['layer19', 'layer27'], 'type': 'diversify', 'mode' : '7', 'prune_propotion': None, 'times': 2}
            })
            pruned_model_path = None
            pruned_model = None
        if FLAGs.Model_2nd == '20_Ternary':
            Model_Path = 'Model/ResNet_Model/ResNet_20_Ternary_99_cifar10_2018.04.13/'
            Model = '160.ckpt'
        if FLAGs.Model_2nd == '20_Binary_1':
            Model_Path = None #'Model/ResNet_Model/ResNet_20_Binary_1_91_cifar10_2018.04.22/'
            Model = None #'200.ckpt'
    
    iter = 0
    while(1):
        error = False
        diversify_layer = copy.deepcopy(diversify_layers[str(iter)])
        
        # Update diversify_layer
        start = int(diversify_layer['layer'][0].split('layer')[-1])
        end = int(diversify_layer['layer'][1].split('layer')[-1])
        layer_ = []
        for layer in range(start, end+1):
            layer_.append('layer'+str(layer))
        diversify_layer.update({'layer': layer_})
        
        # Epoch
        Epoch = FLAGs.Epoch
        if diversify_layer['type']=='restoring':
            Epoch = Epoch
        
        # Pruning/Rebulding iter
        PR_iter = 0
        while(1):
            Global_Epoch = 0
            # Training
            while(1):
                if Global_Epoch < Epoch * 0.9 and FLAGs.Epoch >= 10:
                    epochs_per_eval = 10
                else:
                    epochs_per_eval = FLAGs.epochs_per_eval
                    
                Model_Path_s, Model_s, error, Global_Epoch = utils.run_diversifying(
                    Hyperparameter          = None                  ,               
                    FLAGs                   = FLAGs                 ,
                    Epoch                   = epochs_per_eval       ,
                    Global_Epoch            = Global_Epoch          ,
                    IS_HYPERPARAMETER_OPT   = IS_HYPERPARAMETER_OPT ,
                    error                   = error                 ,
                    Dataset_Path            = Dataset_Path          ,
                    Y_pre_Path              = Y_pre_Path            ,
                    teacher_model_path      = Model_Path            ,
                    teacher_model           = Model                 ,
                    student_model_path      = Model_Path_s          ,
                    student_model           = Model_s               ,
                    pruned_model_path       = pruned_model_path     ,
                    pruned_model            = pruned_model          ,
                    PR_iter                 = PR_iter               ,
                    diversify_layer         = diversify_layer       )
                
                if error:
                    error = False
                    print("... Restart")
                    continue
                    
                # -- Testing --
                test_accuracy = utils.run_testing(
                    Hyperparameter        = None                 ,               
                    FLAGs                 = FLAGs                ,
                    IS_HYPERPARAMETER_OPT = IS_HYPERPARAMETER_OPT,  
                    Dataset_Path          = Dataset_Path         ,
                    testing_model_path    = Model_Path_s         ,
                    testing_model         = Model_s              ,
                    train_target_path     = train_target_path    ,
                    valid_target_path     = valid_target_path    ,
                    test_target_path      = test_target_path     ,
                    train_Y_pre_path      = train_Y_pre_path     ,
                    valid_Y_pre_path      = valid_Y_pre_path     ,
                    test_Y_pre_path       = test_Y_pre_path      ,
                    training_type         = 'diversify'          ,
                    diversify_layers      = diversify_layers     ,
                    is_find_best_model    = False                )
                                
                if Global_Epoch >= Epoch:
                    break
            
            # PR_iter
            if not error:
                PR_iter = PR_iter + 1
            # Prune End
            if PR_iter == int(diversify_layer['times']):
                break

        # iter
        if not error:
            iter = iter + 1
        # The End
        if iter == len(diversify_layers):
            break
        
if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)

