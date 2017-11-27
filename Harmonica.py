import argparse
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import utils
import Model
import csv
import pdb
import sys
import pickle


Dataset = 'mnist' 
class_num = 10
Model_first_name  = 'LeNet' 
Model_second_name = '5' 
Model_Name = Model_first_name + '_' + Model_second_name

#=========#
#   def   #
#=========#
def addNames(curName, curd, targetd, lasti, curID, options, names, ids, N):
	# This method computes the name (and IDs) of the features by recursion
    if curd<targetd:
        for i in range(lasti,N):
            addNames(
				curName = curName + ('*' if len(curName)>0 else '') + options[i], 
				curd    = curd+1, 
				targetd = targetd, 
				lasti   = i+1, 
				curID   = curID+[i], 
				options = options, 
				names   = names, 
				ids     = ids, 
				N       = N)
    elif curd==targetd:
        names.append(curName)
        ids.append(curID)

def readOptions():
	# This method reads option names from options.txt, return as a list
	# options.txt contains names for the options.
	# See option_writer.py for how to generate a options.txt file
	# Of course, you can write your own options.txt file. See options_example.txt for example.
    with open("options.txt") as f:
        content=f.readlines()
    ans=[]
    for line in content:
        s=line.split(",")
        ans.append(s[0])
    return ans

def getBasisValue(input,basis,featureIDList):   
	# Return the value of a basis
    ans=0
	# for every term in the basis
    for (weight,pos) in basis:                  
        term=weight
		# for every variable in the term
        for entry in featureIDList[pos]:        
            term=term*input[entry]
        ans+=term
    return ans

def mask_random_sample(mask_list,N): 
	# mask_list is a list of lists of masks.
	# Initialized with zeros
	current_x    = np.zeros(N)                             
	current_mask = np.zeros(N)
	# need to enumerate all mask levels
	for masks in mask_list:             
		# need to pick one mask from a number of masks
		mask_picked=np.random.randint(len(masks))     
		# picked mask
		mask=masks[mask_picked][0]                    
		for j in range(N):
			# mask has value, not written before
			if (mask[j]!=0) and (current_x[j]==0): 
				# fill in mask value!
				current_x[j]   = mask[j]              
				current_mask[j]= mask[j]
	# If not filled by masks, fill it with random value
	for j in range(N):                                
		if current_x[j]==0:
			current_x[j]=(-1)**np.random.randint(2)
	return current_x, current_mask

def batch_sampling_x(mask_list, n_samples, N):
	# Sequential implementation of batch_sampling.
	# Please modify it if you want parallel implementation, using your favorite tool.
	x=[]
	mask=[]
	for i in range(n_samples):
		current_x, current_mask = mask_random_sample(mask_list,N)
		x.append(current_x)
		mask.append(current_mask)
	return x, mask
	
def batch_sampling_y(x, n_samples, optionNames, currentStage, Continue):
	# Normal
	if not Continue:
		y=[]
		y_ = np.zeros(n_samples)
		is_done = np.repeat(np.array(['X']), n_samples)
		for i in range(n_samples):
			print('\033[1;34;40mcurrentStage\033[0m : {}' .format(currentStage))
			print('\033[1;34;40miteration\033[0m    : {}' .format(i))
			print('-------------------------------------')
			y_tmp = query(x[i])
			y.append(y_tmp)
			# update table
			y_[i] = y_tmp
			is_done[i] = 'O'
			
			file = np.concatenate([np.expand_dims(y_, axis=1), np.expand_dims(is_done, axis=1)], axis=1)
			
			np.savetxt('Harmonica_tmp/' + 'y.csv', file, delimiter=",", fmt='%s')
	# Continue
	else:
		index = 0
		y=[]
		y_ = np.zeros(n_samples)
		is_done = np.repeat(np.array(['X']), n_samples)
		
		with open('Harmonica_tmp/y.csv') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for iter, row in enumerate(reader):
				if row[1] == 'O':
					index = index + 1
					y_tmp = float(row[0])
					y.append(y_tmp)
					y_[iter] = y_tmp
					is_done[iter] = row[1]

		for i in range(index, n_samples):
			print('\033[1;34;40mcurrentStage\033[0m : {}' .format(currentStage))
			print('\033[1;34;40miteration\033[0m    : {}' .format(i))
			print('-------------------------------------')
			y_tmp = query(x[i])
			y.append(y_tmp)
			# update table
			y_[i] = y_tmp
			is_done[i] = 'O'
			
			file = np.concatenate([np.expand_dims(y_, axis=1), np.expand_dims(is_done, axis=1)], axis=1)
			
			np.savetxt('Harmonica_tmp/' + 'y.csv', file, delimiter=",", fmt='%s')
	return y
	
def query(x):
	#=======================#
	#	Global Parameter	#
	#=======================#
	print('\033[1;32;40mMODEL NAME\033[0m :\033[1;37;40m {MODEL_NAME}\033[0m' .format(MODEL_NAME=Model_Name))
	
	IS_HYPERPARAMETER_OPT = True
	IS_TRAINING 		  = True
	IS_TESTING  		  = True
	
	#==========#
	#   Path   #
	#==========#
	# For Loading Dataset
	Dataset_Path = '/home/2016/b22072117/ObjectSegmentation/codes/dataset/' + Dataset
	if Dataset=='ade20k':
		Dataset_Path = Dataset_Path + '/ADEChallengeData2016'
	Y_pre_Path   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/' + Model_first_name + '_Y_pre'
	
	_, _, y, max_parameter = utils.run(
		Hyperparameter			= x,
		# Info
		Dataset 				= Dataset,
		Model_first_name 		= Model_first_name,
		Model_second_name 		= Model_second_name,
		IS_HYPERPARAMETER_OPT 	= IS_HYPERPARAMETER_OPT,
		IS_TRAINING 			= IS_TRAINING,
		IS_TESTING 				= IS_TESTING,
		EPOCH_TIME              = 4,
		# Path
		Dataset_Path			= Dataset_Path,
		Y_pre_Path				= Y_pre_Path,
		train_target_path		= None,
		valid_target_path		= None,
		test_target_path		= None,
		train_Y_pre_path		= None,
		valid_Y_pre_path		= None,
		test_Y_pre_path			= None,
		TESTING_WEIGHT_PATH     = None,
		TESTINGN_WEIGHT_MODEL   = None
	)
	print("\033[1;32;40mScore\033[0m : {}" .format(-y*100 + float(max_parameter) / 3000))
	print("")
	
	return -y*100 + float(max_parameter) / 3000
    
#============#
#    main    #
#============#
"""
Hyperparameter Optimization: A Spectral Approach

	Training Hyperparameter
	|=======================================================================================================|
	| Num|              Type                     |             0                |              1            |
	|=======================================================================================================|
	| 00 | Optimization Method                   | MOMENTUM                     | ADAM                      |
	| 01 | Momentum Rate                         | 0.9                          | 0.99                      |
	| 02 | Initial Learning Rate (1)             | < 0.01                       | >= 0.01                   |
	| 03 | Initial Learning Rate (2)             | < 0.001; <0.1;               | >= 0.001; >= 0.1          |
	| 04 | Initial Learning Rate (3)             | 0.0001; 0.001; 0.01; 0.1;    | 0.0003; 0.003; 0.03; 0.3  |
	| 05 | Learning Rate Drop                    | No                           | Yes                       |
	| 06 | Learning Rate First Drop Time         | Drop by 1/10 at Epoch 40     | Drop by 1/10 at Epoch 60  |
	| 07 | Learning Rate Second Drop Time        | Drop by 1/10 at Epoch 80     | Drop by 1/10 at Epoch 100 |
	| 08 | Weight Decay                          | No                           | Yes                       | 
	| 09 | Weight Decay Lambda                   | 1e-4                         | 1e-3                      |
	| 10 | Batch Size                            | Small                        | Big                       |
	| 11 | Batch Size                            | 32; 128                      | 64; 256                   |
	| 12 | Teacher-Student Strategy              | No                           | Yes                       |
	| 13 | Use Dropout                           | No                           | Yes                       |
	| 14 | Dropout Rate (1)                      | Low                          | High                      |
	| 15 | Dropout Rate (2)                      | 0.05; 0.2                    | 0.1; 0.3                  |
	| 16 | Weight Ternary Epoch                  | 40                           | 60                        |
	| 17 | Activation Quantized Epoch            | 80                           | 100                       |
	|=======================================================================================================|

	Model Hyperparameter : Repeat L Times (L=layer number)                     
	|=======================================================================================================|
	| Num|              Type                     |             0                |              1            |
	|=======================================================================================================|
	| 00 | Weight Ternary                        | No                           | Yes                       |
	| 01 | Activation Quantized                  | No                           | Yes                       |
	| 02 | Batch Normalization                   | No                           | Yes                       |
	| 03 | Shortcut                              | No                           | Yes                       |
	| 04 | Shortcut Distance                     | 1                            | 2                         |
	| 05 | Bottlneck                             | No                           | Yes                       |
	| 06 | Inception                             | No                           | Yes                       |
	| 07 | Dilated (Rate=2)                      | No                           | Yes                       |
	| 08 | Depthwise                             | No                           | Yes                       |
	| 09 | Activation                            | Sigmoid                      | ReLU                      |
	|=======================================================================================================|
"""

parser = argparse.ArgumentParser()
parser.add_argument('-alpha'  , type=float, default=3  , help='weight of the l_1 regularization in Lasso')
parser.add_argument('-nSample', type=int  , default=100, help='number of samples per stage?')
parser.add_argument('-nStage' , type=int  , default=3  , help='number of stages?')
parser.add_argument('-degree' , type=int  , default=3  , help='degree of the features?')
parser.add_argument('-nMono'  , type=int  , default=5  , help='number of monomials selected in Lasso?')
parser.add_argument('-N'      , type=int  , default=98 , help='number of hyperparameters?')
parser.add_argument('-t'      , type=int  , default=1  , help='number of masks picked in each stage?')
opt = parser.parse_args()

learnedFeature = []                 # This is a list of important features that we extracted
bestAnswer = 100000                 # Best answer so far
bestSet  = np.zeros(opt.N)			# Best hyperparameter set so far
bestMask = np.zeros(opt.N)			# Best mask set so far

# List of masks of the value for the fixed important variables
maskList = []
maskList.append(
	[(np.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 
				0. , 0. , -1., 0. , 0. , 0. , -1., -1.,              
				0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , # layer0
				0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , # layer1
				0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , # layer2
				0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , # layer3
				0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , # layer4
				0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , # layer5
				0. , 0. , 0. , 0. , -1., 0. , 0. , 0. , 0. , 0. , # layer6
				0. , 0. , 0. , -1.,  0. , 0. , 0. , 0. , 0. , 0.  # layer7
				]), bestAnswer)]
)


optionNames=readOptions()           # Read the names of options
print("\033[1;32;40mTotal options\033[0m: {TO}" .format(TO = len(optionNames)))
assert(len(optionNames)==opt.N)     # Please be consistent on the number of options names and the number of options.
extendedNames=[]                    # This list stores the names of the features, including vanilla features and low degree features
featureIDList=[]                    # This list stores the variable IDs of each feature
for depth in range(0, opt.degree+1):
    addNames( # This method computes extendedNames and featureIDList
		curName = '', 
		curd    = 0, 
		targetd = depth, 
		lasti   = 0, 
		curID   = [], 
		options = optionNames, 
		names   = extendedNames, 
		ids     = featureIDList,
		N       = opt.N
	)   
print("\033[1;32;40mNumber of features\033[0m: {NoF}" .format(NoF=len(extendedNames)))

featureExtender = PolynomialFeatures(opt.degree, interaction_only=True)   # This function generates low degree monomials. It's a little bit slow though. You may write your own function using recursion.
LassoSolver = linear_model.Lasso(fit_intercept=True, alpha=opt.alpha)     # Lasso solver

continueState = 0
Continue = False

if Continue:
	# currentStage
	with open('Harmonica_tmp/currentStage.txt') as myfile:
		continueState = pickle.load(myfile)
	# x
	with open('Harmonica_tmp/x.txt') as myfile:
		x = pickle.load(myfile)
			
	# mask
	with open('Harmonica_tmp/mask.txt') as myfile:
		mask = pickle.load(myfile)
		
	# maskList
	with open('Harmonica_tmp/maskList.txt') as myfile:
		maskList = pickle.load(myfile)
			
	if continueState!=0:
		# bestMask
		with open('Harmonica_tmp/bestMask.txt') as myfile:
			bestMask = pickle.load(myfile)
			
		# bestSet
		with open('Harmonica_tmp/bestSet.txt') as myfile:
			bestSet = pickle.load(myfile)
			
		# bestAnswer
		with open('Harmonica_tmp/bestAnswer.txt') as myfile:
			bestAnswer = pickle.load(myfile)	
#pdb.set_trace()		
for currentStage in range(continueState, opt.nStage+1):                      # Multi-stage Lasso
	# Save CurrentStage
	with open('Harmonica_tmp/currentStage.txt', 'wb') as myfile:
		pickle.dump(currentStage, myfile)
	
	print("\033[1;34;40mStage\033[0m: {S}" .format(S=currentStage))
	print("Sampling..")
	
	# x, mask
	if not Continue:
		x, mask = batch_sampling_x(maskList, opt.nSample, opt.N)               # Get a few samples at this stage
		with open('Harmonica_tmp/mask.txt', 'wb') as myfile:
			pickle.dump(mask, myfile)
			
		with open('Harmonica_tmp/x.txt', 'wb') as myfile:
			pickle.dump(x, myfile)
	
	# y
	y = batch_sampling_y(x, opt.nSample, optionNames, currentStage, Continue)
	
	
	if np.min(y)<=bestAnswer: # Update the best answer
		bestMask   = mask[np.argmin(y)]
		bestSet    = x[np.argmin(y)]
		bestAnswer = np.min(y)
		
		with open('Harmonica_tmp/bestMask.txt', 'wb') as myfile:
			pickle.dump(bestMask, myfile)
			
		with open('Harmonica_tmp/bestSet.txt', 'wb') as myfile:
			pickle.dump(bestSet, myfile)
			
		with open('Harmonica_tmp/bestAnswer.txt', 'wb') as myfile:
			pickle.dump(bestAnswer, myfile)
			
		
	print('\033[1;32;40mbest answer\033[0m: {BA}' .format(BA = bestAnswer))
	if currentStage==(opt.nStage):
		print("\033[1;32;40mbestMask\033[0m: {B}" .format(B=bestMask))		
		print("\033[1;32;40mbestSet\033[0m: {B}" .format(B=bestSet))
		print("Done!")
		break
	
	for i in range(0,len(y)):
		for basis in learnedFeature:
			y[i]-=getBasisValue(x[i],basis,featureIDList)     # Remove the linear component previously learned
	print("Extending feature vectors..")
	tmp=[]
	for current_x in x:
		tmp.append(featureExtender.fit_transform(np.array(current_x).reshape(1, -1))[0].tolist())   # Extend feature vectors with monomialspython
	x=np.array(tmp)                             # Make it array
	
	print("Running linear regression..")
	LassoSolver.fit(x, y)                       # Run Lasso to detect important features
	coef=LassoSolver.coef_
	index=np.argsort(-np.abs(coef))             # Sort the coefficient, find the top ones
	
	cur_basis=[]
	for i in range(0,opt.nMono):
		cur_basis.append((coef[index[i]],index[i]))         # Add the basis, and its position, only add top nMono                                            
	learnedFeature.append(cur_basis[:])                     # Save these features in learned features
															
	mapped_count = np.zeros(opt.N)                    # Initialize the count matrix
	
	for cur_monomial in cur_basis:                          # For every important feature (a monomial) that we learned
		for cur_index in featureIDList[cur_monomial[1]]:    # For every variable in the monomial
			mapped_count[cur_index]+=1                      # We update its count
	config_enumerate = np.zeros(opt.N)                # Use this array to enumerate all possible configuration (to find the minimum for the current sparse polynomial)
	l=[]                                                    # All relevant variables.
	for i in range(0, opt.N):                         # We only need to enumerate the value for those relevant variables
		if mapped_count[i] > 0:                             # This part can be made slightly faster. If count=1, we can always set the best value for this variable.
			l.append(i)
	
	lists=[]
	for i in range(0, 2**len(l)):  # for every possible configuration
		for j in range(0, len(l)):
			config_enumerate[l[j]]=1 if ((i % (2 ** (j + 1))) // (2 ** j) == 0) else -1
		score=0
		for cur_monomial in cur_basis:
			base=cur_monomial[0]
			for cur_index in featureIDList[cur_monomial[1]]:
				base= base * config_enumerate[cur_index]
			score=score+base
		lists.append((config_enumerate.copy(), score))
	lists.sort(key=lambda x : x[1])
	maskList.append(lists[:opt.t])
	
	with open('Harmonica_tmp/maskList.txt', 'wb') as myfile:
		pickle.dump(maskList, myfile)
		
	Continue = False
	
"""
y=base_hyperband(maskList, 40000,opt.N)                # Run a base algorithm for fine tuning
#y=base_random_search(maskList,100,N)                  # Run random search as the base algorithm
bestAnswer=min(y, bestAnswer)
printSeparator()
print('best answer : ', bestAnswer)
"""
