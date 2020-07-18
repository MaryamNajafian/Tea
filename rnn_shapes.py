import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, SimpleRNN
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model

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
    * i = Input(shape=(T, D))
    * i = Input(shape=(T,1))  # input layer of shape T if D=1
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

#%% Make some data
"""
Things you should automatically know and have memorized
N = number of samples
T = sequence length
D = number of input features
M = number of hidden units
K = number of output units
"""
N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D)

#%% Make an RNN
M = 5 # number of hidden units
i = Input(shape=(T, D))
x = SimpleRNN(units=M)(i) # in RNNs default activation is not None it is tanh
x = Dense(units=K)(x)

model = Model(i,x)

#%% Get the output
Yhat = model.predict(X)
print(Yhat)
#%% See if we can replicate this output
# Get the weights first
model.summary()

# See what's returned
print(model.layers[1].get_weights())

#%% Check their shapes
# Should make sense
# First output is input > hidden
# Second output is hidden > hidden
# Third output is bias term (vector of length M)
a, b, c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)
#%%
Wx, Wh, bh = model.layers[1].get_weights()
# Wx is (DxM), Wh is MxM, bh is is a (M,) and
# Wo and bo are assigned to output layer
Wo, bo = model.layers[2].get_weights()
#%% manual RNN calculation gives us same results as Yhat in simpleRNN
h_last = np.zeros(M)  # initial hidden state
x = X[0]  # the one and only sample
Yhats = []  # where we store the outputs

for t in range(T):
    h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
    y = h.dot(Wo) + bo  # we only care about this value on the last iteration
    Yhats.append(y)

    # important: assign h to h_last
    h_last = h

# print the final output
print(Yhats[-1])


