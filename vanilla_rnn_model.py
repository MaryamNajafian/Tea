"""
RNN for Time series prediction:
It did not perform as autoregressive linear model
This is because RNN has too many parameters and hence flexibility

Linear regression:
    * input shape: 2D array: NxT, output-shape: NxK
    * i = Input(shape=(T,))  # input layer of shape T
    * model.predict(x.reshape(1, -1))[0, 0]
RNN:
    * input shape:3D array: NxTxD or NxTx1, output shape: NxK
    * i = Input(shape=(T,1))  # input layer of shape T
    * model.predict(x.reshape(1, T, 1))[0, 0] because #samples=1, #feature dimensions=1 length=T

Unlike auto-regressive linear model which expects a a 2D array: an NxT array
The vanilla RNN model addresses time series problem using a 3D array of NxTxD
Keep track of the data shapes in RNN
    1-load the data (for RNN data shape: NxTxD)
    2-build/instantiate the model
    3-train the model
    4-evaluate the model
    5-make predictions on unseen test data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

#%% make the original data
# sin(wt)=sin(0.1 * np.arange(200)) wave plus a noise np.random.randn(200)*0.1
series = np.sin(0.1*np.arange(1000)) # added noise :+0.1*np.random.randn(1000)
# plot it
plt.plot(series)
plt.show()


#%% build the dataset to match RNN
# use T past values to predict the next value
# for RNN data shape: NxTxD

T=10
D=1
X=[]
Y=[]

for t in range(len(series)-T):
    x,y = series[t:t+T], series[t+T]
    X.append(x)
    Y.append(y)

#  After np.array(X).reshape(-1,T,1) Now the data should be N x T x D
#  -1 is a wild card; it means make N whatever  dimension necessary
#  such that we have a T, and D=1 dimension in our data
X = np.array(X).reshape(-1,T,1)
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)


#%% Build an autoregressive RNN model

i = Input(shape=(T,1))  # input layer of shape Tx1 for recurrent autoregressive model

# SimpleRNN(units=15) in RNNs default activation is not None it is tanh. The results are not as good as the linear regression
# SimpleRNN(units=15, activation=None) A NN without an activation function is a linear model.
# SimpleRNN(units=15, activation='relu') it copies the last forcasted value over and over not as good model at all
x = SimpleRNN(units=15, activation='relu')(i)
x = Dense(units=1)(x)
model = Model(i,x)
model.compile(
    loss = 'mse',
    optimizer=Adam(lr=0.001)
)

#%% train the RNN
r = model.fit(
X[:-N//2],Y[:-N//2],
epochs=80,
validation_data=(x[-N//2:],Y[-N//2:])
)

#%%
# Plot loss per iteration
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()

#%% IMPORTANT: forecast should occur using predicted targets rather than true targets
validation_target = Y[-N//2:]
validation_predictions = []

# first validation input
last_x = X[-N // 2]  # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, -1, 1))[0, 0]  # 1x1 array -> scalar
    # update the predictions list
    validation_predictions.append(p)

    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
#%%
plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()

