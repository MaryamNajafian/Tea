"""
one-step prediction on stock prices is misleading and also unconventional
more conventional prediction is stock return
R = (V_final - V_initial)/V_initial

we want RNN to take first 10 days of data to predict th 11th day
we explore three methods:
1- first method is wrong and misleading
2- second method is correct but  we see that model doesn't do much except copying the same value over and over again
3-third method is correct and we use all the data: open,high, low, close, volume(D=5) to do a binary classification of whether a stock price gonna go up or down
    forecasting stock prices based on the time series alonre is wrong, and the result shows poor accuracy.
    the act of predicting stock prices from stock prices is a wrong approach and fundamentally flawed
    e.g. emotion of investors, company image in media etc
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Flatten, GlobalMaxPool1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

# %% download dataframes from URLs!
df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')
df.head()
df.tail()

# %% Method 1: Start by doing the WRONG thing - trying to predict the price itself
series = df['close'].values.reshape(-1, 1)  # (1259, 1)
# Normalize the data
# Note: I didn't think about where the true boundary is, this is just approx.
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
# scaler.transform(series) -> (1259,1)
series = scaler.transform(series).flatten()  # -> turns array to vector of form (1259,)

T = 10
D = 1
X = []
Y = []

# Build the dataset
for t in range(len(series) - T):
    x, y = series[t:t + T], series[t + T]
    X.append(x)
    Y.append(y)

X = np.array(X).reshape(-1, T, 1)  # Now the data should be N x T x D
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)  # X.shape (1249, 10, 1) Y.shape (1249,)

# autoregressive RNN model
i = Input(shape=(T, 1))
x = LSTM(units=5)(i)
x = Dense(units=1)(x)
model = Model(i, x)
model.compile(
    loss='mse',
    optimizer=Adam(lr=0.1)
)

# train the RNN
r = model.fit(
    X[:-N // 2], Y[:-N // 2],
    epochs=80,
    validation_data=(X[-N // 2:], Y[-N // 2:]),
)

# Plot loss per iteration
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# One-step forecast using true targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:, 0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

# %% Method2: Multi-step forecast
validation_target = Y[-N // 2:]
validation_predictions = []

# first validation input
last_x = X[-N // 2]  # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    # 1-D array of length T reshape to (1, T, 1)
    p = model.predict(last_x.reshape(1, T, 1))[0, 0]  # 1x1 array -> scalar

    # update the predictions list
    validation_predictions.append(p)

    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()

# calculate returns by first shifting the data
df['PrevClose'] = df['close'].shift(1)  # move everything up 1

# so now it's like
# close / prev close
# x[2] x[1]
# x[3] x[2]
# x[4] x[3]
# ...
# x[t] x[t-1]

df.head()
# then the return is
# (x[t] - x[t-1]) / x[t-1]
df['Return'] = (df['close'] - df['PrevClose']) / df['PrevClose']
# Now let's try an LSTM to predict returns
df['Return'].hist()

series = df['Return'].values[1:].reshape(-1, 1)

# Normalize the data
# Note: I didn't think about where the true boundary is, this is just approx.
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()

#%% Method 3
### build the dataset
# let's see if we can use T past values to predict the next value
T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T, 1) # Now the data should be N x T x D
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

### try autoregressive RNN model
i = Input(shape=(T, 1))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.01),
)

# train the RNN
r = model.fit(
  X[:-N//2], Y[:-N//2],
  epochs=80,
  validation_data=(X[-N//2:], Y[-N//2:]),
)

# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# One-step forecast using true targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

# Multi-step forecast
validation_target = Y[-N // 2:]
validation_predictions = []

# first validation input
last_x = X[-N // 2]  # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, T, 1))[0, 0]  # 1x1 array -> scalar

    # update the predictions list
    validation_predictions.append(p)

    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
#%% Method 3 - Now turn the full data (5 columns in df) into numpy arrays
# lets predict whther the price go up or down (binary classification)
# we will have a N x T x D signal where D=5 and T=10

# Not yet in the final "X" format!
input_data = df[['open', 'high', 'low', 'close', 'volume']].values
targets = df['Return'].values
# Now make the actual data which will go into the neural network
T = 10 # the number of time steps to look at to make a prediction for the next day
D = input_data.shape[1] #5
N = len(input_data) - T # (e.g. if T=10 and you have 11 data points then you'd only have 1 sample)
# normalize the inputs
Ntrain = len(input_data) * 2 // 3
# normalize the data specially because
# volume column is much bigger than price column
scaler = StandardScaler()
scaler.fit(input_data[:Ntrain + T - 1])
input_data = scaler.transform(input_data)

# Setup X_train and Y_train
X_train = np.zeros((Ntrain, T, D))
Y_train = np.zeros(Ntrain)

for t in range(Ntrain):
  X_train[t, :, :] = input_data[t:t+T]
  Y_train[t] = (targets[t+T] > 0)

# Setup X_test and Y_test
X_test = np.zeros((N - Ntrain, T, D))
Y_test = np.zeros(N - Ntrain)

for u in range(N - Ntrain):
  # u counts from 0...(N - Ntrain)
  # t counts from Ntrain...N
  t = u + Ntrain
  X_test[u, :, :] = input_data[t:t+T]
  Y_test[u] = (targets[t+T] > 0)
# make the RNN
i = Input(shape=(T, D))
x = LSTM(50)(i)
x = Dense(1, activation='sigmoid')(x)
model = Model(i, x)
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr=0.001),
  metrics=['accuracy'],
)
# train the RNN
r = model.fit(
  X_train, Y_train,
  batch_size=32,
  epochs=300,
  validation_data=(X_test, Y_test),
)
# plot the loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()