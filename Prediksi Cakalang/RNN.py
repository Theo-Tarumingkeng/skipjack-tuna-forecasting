# Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import pickle

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Load the dataset
dataset = pd.read_csv('Hasil tangkapan & variabel lingkungan.csv', header=0, index_col=0)
dataset.set_index('datetime', inplace=True)  # Setting 'datetime' as index
dataset.drop(dataset.columns[[1, 2, 3, 4, 11, 13, 15]], axis=1, inplace=True)
values = dataset.values

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names

	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# integer encode direction
#encoder = LabelEncoder()
#values[:,7] = encoder.fit_transform(values[:,7])
#values[:,9] = encoder.fit_transform(values[:,9])
#values[:,11] = encoder.fit_transform(values[:,11])

# ensure all data is float
values = values.astype('float32')

# frame as supervised learning
reframed = series_to_supervised(values, 1, 1)

values_reframed = reframed.values

# normalize data before splitting it
scaler = MinMaxScaler(feature_range=(0,1))
values_reframed = scaler.fit_transform(values_reframed)

# split into train and test sets
n_train = 708 #885
train = values_reframed[:n_train, :]
test = values_reframed[n_train:, :]

# split into input and outputs(target)
train_input, train_target = train[:, :-1], train[:, -1]
test_input, test_target = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_input = train_input.reshape((train_input.shape[0], 1, train_input.shape[1]))
test_input = test_input.reshape((test_input.shape[0], 1, test_input.shape[1]))

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_input.shape[1], train_input.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
model.fit(train_input, train_target, epochs=100, batch_size=72, validation_data=(test_input, test_target), verbose=2, shuffle=False)
#history = model.fit(train_input, train_target, epochs=100, batch_size=72, validation_data=(test_input, test_target), verbose=2, shuffle=False)

print(train_input.shape, train_target.shape, test_input.shape, test_target.shape)

# make pickle file of our model
pickle.dump(model, open("RNN model.pkl", "wb"))