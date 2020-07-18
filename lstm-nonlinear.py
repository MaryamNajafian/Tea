"""
* ways to forecast
    * one-step forecasting: it is artificially good
        * classical models likeArima build a 1 step predictor, the itteratively apply it to predit multiple steps
    * multi-step forecasting: iteratively builds a multi-step forecast using
        * model's own prediction can lead to poor results, even on simple  problems like sine wave
        * e.g. a NN with multiple outputs: a Dense(units=12) can  predict 12 steps ahead
    * Naive forecast: just predict the last value
        * stock prices closely follow a random walk so a naive forecast is the best
    * Implementation: in TF wew have constant length time series so our data should be of form NxTxD array
"""


#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Dense,Flatten, SimpleRNN,GRU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
#%%
# make data x(t) =sin((w.t)^2)
series = np.sin((0.1*np.arange(400)**2))
# plot it
plt.plot(series)
plt.show()
# build dataset
T=10
D=1
X=[]
Y=[]
#%%
for t in range(len(series) - T):
    x,y = series[t:t+T],series[t+T]
    X.append(x)
    Y.append(y)

X = np.array(X).reshape(-1,T) # make it NxT; -1 is the wild card to replace N
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape,"Y.shape", Y.shape)
#%%
## try autoregressive linear model
# train the RNN
# fit(), will train the model by slicing the data into "batches"
# of size "batch_size", and repeatedly iterating over the entire
# dataset for a given number of "epochs".
# We pass some validation for
# monitoring validation loss and metrics
# at the end of each epoch

i = Input(shape=(T,))
x = Dense(units=1,activation=None)(i)
model = Model(inputs=i,outputs=x)
model.compile(
    loss='mse',
    optimizer=Adam(lr=0.01)
)
r = model.fit(
    x=X[:-N//2],y=Y[:N//2],
    epochs=80,
    validation_data=(X[-N//2:],Y[-N//2:])
)
#%%
# Plot loss per iteration
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

#%%
# one-step forecast using true targets
#note: even the one-step forecast fails

outputs=model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title("Linear Regression Predictions")
plt.legend()
plt.show()

#%%
# This does the same thing as above
# One-step forecast using true targets

validation_target = Y[-N // 2:]
validation_predictions = []

# index of first validation input
i = -N // 2

while len(validation_predictions) < len(validation_target):
    p = model.predict(X[i].reshape(1, -1))[0, 0]  # 1x1 array -> scalar
    i += 1

    # update the predictions list
    validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()

#%%
# Multi-step forecast
validation_target = Y[-N//2:]
validation_predictions =[]

# first validation input
last_x = X[-N//2] # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    p=model.predict(last_x.reshape(1,-1))[0,0] # 1x1 array -> scalar
    # update the predictions list
    validation_predictions.append(p)
    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction ')
plt.legend()
plt.show()

# #%% #Build an autoregressive linear model
# i = Input(shape=(T,))
# x = Dense(units=1,activation=None)(i)
# model = Model(inputs=i,outputs=x)
# model.compile(
#     loss='mse',
#     optimizer=Adam(lr=0.01)
# )
# r = model.fit(
#     x=X[:-N//2],y=Y[:N//2],
#     epochs=80,
#     validation_data=(X[-N//2:],Y[-N//2:])
# )
#%%
# #Build an autoregressive simple RNN model
# print(X.shape)# (N=390,T=10)
# X = X.reshape(-1,T,1) ## make it N x T x D: -1 is the wild card to replace N
# i = Input(shape=(T,1))  # input layer of shape Tx1 for recurrent autoregressive model
# x = SimpleRNN(units=10, activation='relu')(i)
# x = Dense(units=1)(x)
# model = Model(inputs=i,outputs=x)
# model.compile(
#     loss = 'mse',
#     optimizer=Adam(lr=0.001)
# )

# # train the RNN
# r = model.fit(
#   X[:-N//2], Y[:-N//2],
#   batch_size=32,
#   epochs=200,
#   validation_data=(X[-N//2:], Y[-N//2:]),
# )
#%%
print(X.shape)# (N=390,T=10)
D=1
## Now try RNN/LSTM model
X = X.reshape(-1,T,1) ## make it N x T x D: -1 is the whild card to replace N
# make RNN/LSTM
i=Input(shape=(T,D))
x=LSTM(units=10)(i)
x=Dense(units=1)(x)
model = Model(inputs=i,outputs=x)
model.compile(loss='mse',
              optimizer=Adam(lr=0.05)
              )

# train the RNN
r = model.fit(
  X[:-N//2], Y[:-N//2],
  batch_size=32,
  epochs=200,
  validation_data=(X[-N//2:], Y[-N//2:]),
)


plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.show()
#%%
# one-step forecasting
outputs=model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]
plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title('many-to-one RNN')
plt.legend()
plt.show()
#%%
# Multi-step forecast
forecast = []
input_ = X[-N//2]
while len(forecast) < len(Y[-N//2:]):
  # Reshape the input_ to N x T x D
  f = model.predict(input_.reshape(1, T, 1))[0,0]
  forecast.append(f)
  # make a new input with the latest forecast
  input_ = np.roll(input_, -1)
  input_[-1] = f

plt.plot(Y[-N//2:], label='targets')
plt.plot(forecast, label='forecast')
plt.title("RNN Forecast")
plt.legend()
plt.show()
