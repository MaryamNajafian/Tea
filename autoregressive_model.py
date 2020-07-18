"""
auto-regressive linear model to address a time series problem
linear regression expects a a 2D array so we have to pass an NxT array
"""
# %%
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make the original data is a sin(wt)=sin(0.1 * np.arange(200)) wave plus a noise np.random.randn(200)*0.1
series = np.sin(0.1 * np.arange(200))  + np.random.randn(200)*0.1

# plot it
plt.plot(series)
plt.show()

# let's see if we can use T past values to predict the next value
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t + T]
    X.append(x)
    y = series[t + T]
    Y.append(y)

#  After np.array(X).reshape(-1,T) Now the data should be N x T
#  -1 is a wild card; it means make N whatever dimension necessary
#  such that we have a T dimension in our data
X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
# X is 2D array of size Nx(D=T=10)
# Y is 1D array of size N
print("X.shape", X.shape, "Y.shape", Y.shape)

#%% try auto-regressive linear model

# we make prediction on the i+1 given
# past i items some of them were
# predicted during previous steps
i = Input(shape=(T,))  # input layer of shape T
x = Dense(units=1)(i)  # Dense layer with output and no activation
model = Model(i, x)
model.compile(
    loss='mse',
    optimizer=Adam(lr=0.1),
)

# train the model on half the data and test on the other half
r = model.fit(
    X[:-N // 2], Y[:-N // 2],
    epochs=80,
    validation_data=(X[-N // 2:], Y[-N // 2:]),
)
#%%
# Plot loss per iteration
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# %%  for-casting
# Forecast future values (use only self-predictions for making future predictions)

validation_target = Y[-N // 2:]
validation_predictions = []

# we should not use the true value of X to predict value of Y
# we should use predicted value of X to predict Y

# first validation input
# we update last_x with our latest for-casted prediction
last_x = X[-N // 2]  # 1-D array of length T=10 from second half

# reshape the input and grab the output at [0,0] index to get an scalar valuer for p
while len(validation_predictions) < len(validation_target):
    # model.predict() input should be 2D array and it returns a 2D NxK output
    p = model.predict(last_x.reshape(1, -1))[0, 0]
    # model.predict(last_x.reshape(1, -1)) output is a 1x1 array of length T=10 ->
    # to get the first element of the array to become scalar we use [0,0]
    # last_x.reshape(1, -1):
    # array([[0.10560891, 0.299045, 0.28397024, 0.50397182, 0.44337642,
    #        0.69766665, 0.57517558, 0.87510037, 0.67028868, 1.03343046]])
    # predicted_val=model.predict(last_x.reshape(1, -1))
    # array([[0.72079164]], dtype=float32)
    # predicted_val.reshape(1, -1)[0, 0]:
    # 0.72079164

    # update the predictions list
    validation_predictions.append(p)

    # make the new input
    # shift or roll to the left by 1, and left most value loops back around to the right but
    # it gets replaced with latest prediction p and we update the value last_x with our latest prediction
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
#%%
plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()


