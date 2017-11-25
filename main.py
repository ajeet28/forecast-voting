import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
import argparse
import load_data as ld
from keras.callbacks import ModelCheckpoint

def model(input_dim, input_length, output_dim):
	'''
	LSTM model
	input : feature dimension, sequence length
	output : autoencoder model
	'''
	model = Sequential()
	model.add(LSTM(64, input_dim=input_dim, input_length=input_length, return_sequences=True))
	model.add(LSTM(128, return_sequences=True))
	model.add(LSTM(output_dim, return_sequences=True))
	model.compile(loss = 'mse', optimizer = 'RMSprop', metrics=['acc','mae'])
	return model


def train(model, data, args):
	'''
	Training model
	input : (model, data, arguments)
	output :  trained model
	'''
	(X_tr, Y_tr), (X_val, Y_val) = data
	checkpoint1             =   ModelCheckpoint('model_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')
	callbacks_list          =   [checkpoint1]
	hist = model.fit(X_tr, Y_tr, nb_epoch=args.epochs, batch_size=args.batch_size, validation_data=(X_val, Y_val), callbacks = callbacks_list)
	return hist

def test(model, data):
	'''
	Predicts MAE score
	input : (model, data)
	output : MAE score
	'''
	(X_test, Y_target) = data
	Y_pred = model.predict(X_test)
	MAE = 0
	for i in range(len(Y_target)):
		MAE += np.sum(np.absolute(Y_pred[i][54]-Y_target[i]))

	MAE /= len(Y_target)
	MAE /= 198
	return MAE

if __name__ == "__main__":

	# setting the hyper parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
	parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
	parser.add_argument('--is_training', default=1, type=int, help='Training(1) or testing(0)')
	parser.add_argument('--data_path', default='data/',help='Path to data folder')
	args = parser.parse_args()
	
	#load data
	(X_tr, Y_tr), (X_val, Y_val), (X_test, Y_target) = ld.load()

	input_dim, input_length, output_dim = 244, 55, 198
	#define model
	model = model(input_dim, input_length, output_dim)
	
	# train or test
	if args.is_training:
		hist = train(model=model, data=((X_tr, Y_tr), (X_val, Y_val)), args=args)
	else:  # as long as weights are given, will run testing
		model.load_weights('model_weights.h5')
		MAE = test(model=model, data=((X_test, Y_target)))
		print('Mean Absolute error is: ', MAE)



