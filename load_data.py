import numpy as np
import csv
import os

#data files
trainFolder	=	'data/train/'
testFolder	=	'data/test/'
trainLabelFile	=	'data/trainLabels.csv'
testLabelFile	=	'data/testLabels.csv'

#Reading data
trainFiles	=	os.listdir(trainFolder)
testFiles	=	os.listdir(testFolder)


def load():
	## Reading train data
	train={}

	for filename in trainFiles:
		reader	=	csv.reader(open(trainFolder + filename, 'rb'))
		counter = 0
		tmp_data = []
		
		for row in reader:
		        if counter == 0:
		            counter += 1
		            continue
		        tmp_data.append(row)
		train[filename]    =   np.array(tmp_data)

	## Reading test data
	test={}
	for filename in testFiles:
		reader	=	csv.reader(open(testFolder + filename, 'rb'))
		counter = 0
		tmp_data = []
		
		for row in reader:
		        if counter == 0:
		            counter += 1
		            continue
		        tmp_data.append(row)
		test[filename]    =   np.array(tmp_data)

	#Reading train labels       
	reader	=	csv.reader(open(trainLabelFile,'rb'))
	trainLabels=[]
	counter	=	0
	for row in reader:
		if counter == 0:
		    counter += 1
		    continue
		trainLabels.append(row)
	trainLabels    =   np.array(trainLabels)

	#reading test labels
	reader	=	csv.reader(open(testLabelFile,'rb'))
	testLabels=[]
	counter	=	0
	for row in reader:
		if counter == 0:
		    counter += 1
		    continue
		testLabels.append(row)
	testLabels    =   np.array(testLabels)

	num_samples = len(train)
	X_train1 = np.zeros((num_samples, 55, 244 ))
	X_train2 = np.zeros((num_samples, 55, 442 ))
	Y = np.zeros((num_samples, 55, 198))

	for item, jitem in train.iteritems():
		base = os.path.basename(item)
		Y[int(os.path.splitext(base)[0])-1, :, :] = jitem[:,:198]
		X_train1[int(os.path.splitext(base)[0])-1, :, : ] = jitem[:,198:]
		X_train2[int(os.path.splitext(base)[0])-1, :, : ] = jitem[:,:]

	Y_train = np.zeros((num_samples, 55, 198))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]-1):
			Y[i, j, : ] = Y[i, j+1, :]
		Y_train[i, 54, : ] = trainLabels[i, 1:]

	num_test_samples = len(test)

	X_test = np.zeros((num_test_samples, 55, 244 ))
	Y_target = np.zeros((num_test_samples, 198))

	for item, jitem in test.iteritems():
		base = os.path.basename(item)
		X_test[int(os.path.splitext(base)[0])-151, :, : ] = jitem[:,198:]

	for i in range(Y_target.shape[0]):
		Y_target[i, : ] = testLabels[i, 1:]

	#spliting training into validation set
	X_tr = X_train1[:100,:,:]
	Y_tr = Y_train[:100,:,:]
	X_val = X_train1[100:,:,:]
	Y_val = Y_train[100:,:,:]
	
	return (X_tr, Y_tr), (X_val, Y_val), (X_test, Y_target) 

