#%%
import tensorflow as tf
print(tf.__version__)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv", filename="moore.csv")

data = pd.read_csv('moore.csv', header=None).values

#np.array().reshape(-1,N): gives you a matrix with N columns and length/N rows
X=data[:,0].reshape(-1,1)    #(-1,1) #make it 2-D array of size NxD rather than of !D array of length N
Y=data[:,1] # 1D array of length N

plt.scatter(X,Y)
plt.show()

#%%
# since we want a linear model lets take the log
Y = np.log(Y)
plt.scatter(X,Y)
plt.show()
#%%
# pre-processing
# Let's also center the X data so the values are not too large
# We could scale it too but then we'd have to reverse the transformation later
X = X - X.mean()

#%%
# Now create our Tensorflow model
# Create a TF model with an input and output layers both have dimensionality one and
# we have a dense layer and since its a linear regression there is NO activation function!
# for other model we use activation: tf.keras.layers.Dense(1,activation = 'sigmoid') where output-size =1
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1, activation=None)
])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')


# model.compile(optimizer='adam', loss='mse')


# learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

plt.plot(r.history['loss'],label='loss')
#%%
# Get the slope of the ,ine
# the slope of the line us related to the 2X rate of transmitter count
# Here there is only 1 layer, the "input" layer doesn't count
# the only job the input layer is to keep track of the input size
print(model.layers)
print('\n')
print(model.layers[0].get_weights())
# model.layers[0].get_weights() returns [w, b] = [array([[0.33895805]], dtype=float32), array([17.783209], dtype=float32)]
# given that D=input size and M=output size
# W.shape =(D,M)
# b.shape = (M,)
# here M and D are of size 1

# the slope of the line
a = model.layers[0].get_weights()[0][0,0]
#%%
print("Time to double:", np.og(2) / a)
# If you know the analytical solution
# np.array(X).shape = (162, 1)
# np.array(X).flatten().shape = Return a copy of the array collapsed into one dimension = (162,)
X = np.array(X).flatten()
Y = np.array(Y)
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator
print(a, b)
print("Time to double:", np.log(2) / a)

#%%
# Making Predictions
Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)

# Manual calculation

# Get the weights
w, b = model.layers[0].get_weights()

# Reshape X because we flattened it again earlier
X = X.reshape(-1, 1)

# (N x 1) x (1 x 1) + (1) --> (N x 1)
Yhat2 = (X.dot(w) + b).flatten()

# Don't use == for floating points
np.allclose(Yhat, Yhat2)

#%%
"""
Our original model for exponential growth is:

$$ C = A_0 r^t $$

Where $ C $ is transistor the count and $ t $ is the year.

$ r $ is the rate of growth. For example, when $ t $ goes from 1 to 2, $ C $ increases by a factor of $ r $. When $ t $ goes from 2 to 3, $ C $ increases by a factor of $ r $ again.

When we take the log of both sides, we get:

$$ \log C = \log r * t + \log A_0 $$

This is our linear equation:

$$ \hat{y} = ax + b $$

Where:

$$ \hat{y} = \log C $$
$$ a = \log r $$
$$ x = t $$
$$ b = \log A_0 $$

We are interested in $ r $, because that's the rate of growth. Given our regression weights, we know that:

$$ a = 0.34188038 $$

so that:

$$ r = e^{0.34188038} = 1.4076 $$

To find the time it takes for transistor count to double, we simply need to find the amount of time it takes for $ C $ to increase to $ 2C $.

Let's call the original starting time $ t $, to correspond with the initial transistor count $ C $.

Let's call the end time $ t' $, to correspond with the final transistor count $ 2C $.

Then we also have:

$$ 2C = A_0 r ^ {t'} $$

Combine this with our original equation:

$$ C = A_0 r^t $$

We get (by dividing the 2 equations):

$$ 2C/C = (A_0 r ^ {t'}) / A_0 r^t $$

Which simplifies to:

$$ 2 = r^{(t' - t)} $$

Solve for $ t' - t $:

$$ t' - t = \frac{\log 2}{\log r} = \frac{\log2}{a}$$


Important note! We haven't specified what the starting time $ t $ actually is, and we don't have to since we just proved that this holds for any $ t $.
"""

