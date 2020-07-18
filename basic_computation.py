import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%%
c = tf.constant(4.)
c =c + 1
print(c)

#%%
# A list is mutable
a = [1,2,3]
a[0] = 5
print(a)

# Now Tensorflow variables
a = tf.Variable(5.)
b = tf.Variable(3.)
print(a * b)

#%%
a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.sqrt(a**2 + b**2)
print("c:", c)

# if you use Python 3 f-strings it will print
# the tensor as a float
print(f"c: {c}")

# Get the Numpy version of a Tensor
c.numpy()

print(type(c.numpy()))

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print(f"b: {b}")
c = tf.tensordot(a, b, axes=[0,0])# sum over 0 access  axes=[0,0]
print(f"c: {c}")

a.numpy().dot(b.numpy())

A0 = np.random.randn(3, 3)
b0 = np.random.randn(3, 1)
c0 = A0.dot(b0)
print(f"c0: {c0}")

A = tf.constant(A0)
b = tf.constant(b0)
c = tf.matmul(A, b)
print(f"c: {c}")

# Broadcasting
A = tf.constant([[1,2],[3,4]])
b = tf.constant(1)
C = A + b
print(f"C: {C}")

# Element-wise multiplication
A = tf.constant([[1,2],[3,4]])
B = tf.constant([[2,3],[4,5]])
C = A * B
print(f"C: {C}")

#%%
# First, what is the difference between mutable and immutable?

# A tuple is immutable
# This should result in an error
a = (1,2,3)
a[0] = 5 #TypeError


# A list is mutable
a = [1,2,3]
a[0] = 5
print(a)

# Now Tensorflow variables
a = tf.Variable(5.)
b = tf.Variable(3.)
print(a * b)

# Because it's a variable, it can be updated
a = a + 1
print(a)

# Variables and constants
c = tf.constant(4.)
print(a * b + c) # tf.Tensor(22.0, shape=(), dtype=float32)


#%%
# Let's demonstrate a simple optimization problem
# L(w) = w**2

w = tf.Variable(5.)


# Now, let us define a loss function
def get_loss(w):
    return w ** 2


# Use "gradient tape" to record the gradients
def get_grad(w):
    with tf.GradientTape() as tape:
        L = get_loss(w) # calculate the loss
    g = tape.gradient(L, w) # Get the gradient of L with respect to w
    return g

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1) # Define an optimizer to use when training the model

# Store the losses
losses = []

# Perform gradient descent
for i in range(50):
    g = get_grad(w)
    optimizer.apply_gradients(zip([g], [w]))
    losses.append(get_loss(w))


plt.plot(losses)
print(f"Final loss: {get_loss(w)}")

#%%
# Let's do the same thing again, but manually

w = tf.Variable(5.)

# Store the losses
losses2 = []

# Perform gradient descent
for i in range(50):
  # This is doing: w = w - 0.1 * 2 * w
  # But we don't want to create a new Tensor
  # w.assign(w - learning_rate*grad)
  w.assign(w - 0.1 * 2 * w)
  losses2.append(w ** 2)

plt.plot(losses, label="losses tf")
plt.plot(losses2, label="losses manual")
plt.legend()
#%%
# Define linear regression model

class LinearRegression(tf.keras.Model):
  def __init__(self, num_inputs, num_outputs):
    super(LinearRegression, self).__init__()
    self.W = tf.Variable(
        tf.random_normal_initializer()((num_inputs, num_outputs)))
    self.b = tf.Variable(tf.zeros(num_outputs))
    self.params = [self.W, self.b]

  def call(self, inputs): # estimates forward direction: returns w*x+b
    return tf.matmul(inputs, self.W) + self.b


# Create a dataset
N = 100
D = 1
K = 1
X = np.random.random((N, D)) * 2 - 1
w = np.random.randn(D, K)
b = np.random.randn()
Y = X.dot(w) + b + np.random.randn(N, 1) * 0.1

plt.scatter(X, Y)

# Cast type, otherwise Tensorflow will complain
# because numpy creates double by default but tensorflow expects/creates floats
X = X.astype(np.float32)
Y = Y.astype(np.float32)

# Define the loss
def get_loss(model, inputs, targets):
  predictions = model(inputs)
  error = targets - predictions
  return tf.reduce_mean(tf.square(error))


# Gradient function
def get_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        # calculate the loss
        loss_value = get_loss(model, inputs, targets)

    # return gradient
    return tape.gradient(loss_value, model.params)

# Create and train the model
model = LinearRegression(D, K)

# Print the params before training
print("Initial params:")
print(model.W)
print(model.b)

# Store the losses here
losses = []

# Create an optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

# Run the training loop
for i in range(100):
    # Get gradients
    grads = get_grad(model, X, Y)

    # Do one step of gradient descent: param <- param - learning_rate * grad
    optimizer.apply_gradients(zip(grads, model.params))

    # Store the loss
    loss = get_loss(model, X, Y)
    losses.append(loss)

plt.plot(losses)

x_axis = np.linspace(X.min(), X.max(), 100)
y_axis = model.predict(x_axis.reshape(-1, 1)).flatten()

plt.scatter(X, Y)
plt.plot(x_axis, y_axis)

print("Predicted params:")
print(model.W)
print(model.b)

print(f"True params:{w}, {b}")
