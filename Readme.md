## Installations
`conda create -n py3tf2 python=3 tensorflow=2`

## Steps of ML
1-load the data
    * X,Y
    * Typically Pandas for structured data unless the data is unstructured
    * Split the data into Train/Dev/Test sets
    * It is important to normalize/standardize data before passing it into linear/logistic regression
        * subtracting the data from the mean and dividing by std deviation: z  = (x - mu)/std
    * Batch normalization: if we have a sequence of Dense layers normalizing/standardizing operations is done only on input data, and only first layer sees normalized data
        * however after being transformed by Dense layer its no longer normalized
        * so we use batch normalization to make the data at every layer normalized
            * so instead of: Dense > Dense > , ... 
            * we use:        BatchNorm > Dense > BatchNorm > Dense, ...
            * find current batch mean, and std 
                * `z = (x - batch_mean)/batch_std` and
                * `y=z.gamma+beta` beta and gamma are learnt via gradient descent
        * Batch Norm as Regularization: because each batch of data is different, you will get different batch mean and std (they are not true meann/std of whole dataset)
            * This is essentially noise, and using noise during training makes the NN impervious to noise rather than fitting to the noise (overfiting)
            * Batch Norm is used for dense and conv layers
                *  conv > batch_norm > conv > batch_norm >...> flatten > dense > dense
            * Batch normalization helps avoid overfiting and it accelerates NN training by reducing internal covariance shift
    * Data augmentations: virtually adding more data (e.g. orienting images)

2- Build/instantiate the model
    * OOP
    * TF 2 standard is Keras API, very similar to Scikit-learn
    * Sequential Dense layers ending with classification or regression

3- Train(fit) the model
    * model.fit(X,Y)
    * train the model to minimize Cost/loss/Error function
    * Use Gradient descent to minimize the cost d(J(w))/d(w)=0

4- Evaluate the model on train, validation, and test set
    * see if the data is over fitting and under fitting
    * plot the cost per iteration 
    * Decision rule: 
    * a: is activation and it is of form a = w^T.x + b = sum(w_d.x_d+b) for d between 1 and D

    y_hat = u(a) : u is the step function
    y_hat = sigmoid(a) : sigmoid rather than step function 
    * both `u` and `sigmoid` are called activation function
    * we want a smooth differentiable sigmoid function
        * if a > 0 : predict class 1
        * if a < 0 : predict class 0
    * In y_hat = sigmoid(a) we can interpret the output as probability
    
5- Make model predictions
    * model.predict(X)
    
## Models
* Regression: We want to best fit line/plane/hyperplane or a curve to some data points : y_hat = wx+b 
* Classification: In classification we use the line/plane/hyperplane to separate data points of different classes

* Linear Regression: Dense
* Binary Logistic Regression: Dense> Sigmoid
* Multi-class Logistic Regression: Dense> Softmax
* ANN Regression: Dense> Dense
* ANN Binary Classification: Dense> Dense> Sigmoid
* ANN Multiclass Classification: Dense> Dense> Softmax


 
* We use `Linear regression` for `regression`. 
    * y_hat = w^T.x + b
    * J = MSE = 1/N (sum(y_i -y_i_hat)^2) 
    * Minimize the cost 
        * closed form solution: set gradient(J)/w = 0 --> Only in linear regression we can use it
        * Gradient Descent (GD) for logistic regression onward we use GD for every epoch: w = w - learning rate * gradient(J)/w (same for b)
            * take small steps in the direction of the epoch (step size is shown by etha which is referred as learning rate which determine how fast or slow you want to train your model) 
            * how to choose etha: check loss per iteration
                * high learning rate: loss shoot-off to infinity (you are overshooting the  as steps are too large)
                * low learning rate: you will see a very shallow curve (you have to stop at a suboptimal point as learning is too slow)
                
* We use `Logistic regression (neuron)` for `classification`.
    *  p(y=1|x) = sigmoid(w^T.x + b) 
    
   

### Linear regression
we find the line of best fit 
* y_hat = w^T.x + b

### Logistic regression
when we apply the sigmoid function on top of a linear function it is called logistic regression
* sigmoid function is S shaped function between 0 and 1
    * p(y=1|x) = sigmoid(w^T.x + b) 

### binary vs multi-class classification

* p(y=k|x) for multi-class classification: softmax(((...sigmoid(w^T.x + b))))
    * softmax(a_k) = p(y=k|x) = exp(a_k)/sum(exp(a_j)) for j in [1,K]
    * sum(p(y=k|x))  for k in [1,K] = 1
    * in TF softmax is considered n activation function but unlike sigmoid its not meant for hidden layers
            
* p(y=1|x) for binary classification: sigmoid(((...sigmoid(w^T.x + b)))) = sigmoid(sum(w_d.x_d + b)) for d between 1 and D
    * D = input size 
    * M = output size
    * W.shape = (D,M)
    * b.shape = (M,)
    * here M and D are of size 1
    * sigmoid is [0,1] : sigmoid(i) = 1/(1+exp(-i))
    * tanh is [-1,1]   : tanh(i) =  (exp(2*i)-1)/(exp(2*i)+1)
    
    * major problem with sigmoid and tanh is the vanishing gradient problem
        * sigmoid: Between 0 and 1 centered around 0.5: sigmoid(a) = 1/(1+exp(-a)) 
        * tanh: Between -1 and 1 centered around 0: tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
        * tanh and sigmoid: both are problematic activation functions: they both cause vanishing gradient problem
            * we start by finding gradient  of the loss function with respect to the weight 
            * Output = sigmoid(sigmoid(sigmoid...)), the deeper a NN, the more terms have to be multiplied in gradient (chain rule)
            * we end up multiplying by the derivative of the sigmoid over and over again
            * derivative of the sigmoid is very flat (max number of 0.25)
            * when you multiply the numbers that are very small you get a smaller number (0.25^5) ~ 0.00001
            * the deeper you go in the network the smaller the gradient gets and it vanishes once it is 0
        * we should not use activation functions with vanishing gradients
            * you can use the following activation functions
            * Softplus and ReLU, ELU have vanishing gradient problem on the left(but it doesn't cause an issue) 
              because they are not centered around 0
            * ReLU(a) = max(0,a)
            * LReLU
            * ELU:Exponential Linear Unit negative values are possible and mean can be 0 (unlike RelU)
            * Softplus: f(a) = log(1+e^a)
                   
    
 #### implementation in Keras:
1-instantiate the model
* keras 'sequential' object means that each layer mentioned inside the `list` occurs sequentially, one after the other 
 `model = tf.keras.models.Sequential(list)` where `list` is 
 `list = [tf.keras.layer.Input(shape=(D,)),tf.keras.layers.Dense(outputsize=1,activation = 'sigmoid') ]` 
    * Input object tells keras the size of your input object 
        *`tf.keras.layer.Input(shape=(D,))`
    * Hidden layer:
        * for structured data: `tf.keras.layers.Dense(units=128, input_shape=(2,), activation='relu')`
        * for unstructured image data:     
              `tf.keras.layers.Flatten(input_shape= (28,28))`
              `tf.keras.layers.Dense(128, activation='relu')`
    * outputsize is represented by `units`
        * Regression: 
            * `tf.keras.layers.Dense(units=1, activation = None)`
        * binary classification: keras knows the size of input object, so we only specify size of output, activation function:
            * `tf.keras.layers.Dense(units=1, activation = 'sigmoid')`
        * multi-class classification (K classes): 
            * `tf.keras.layers.Dense(units=K, activation = 'softmax')`

2- Train/Fit the model using a cost/loss function
    * `model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`
    * To minimize the cost the general strategy to find gradient descent e.g we can use `adam` or `SGD` optimizer
    * To evaluate the model on train, validation, and test set you can use `accuracy`
    * Learning rate scheduling: decrease the learning rate depending on the epoch number
    * Loss: we set the loss according to the problem(for linear regression: loss: `MSE`; for logistic regression: `binary cross entropy`)

3- after compiling we should use `fit` function to complete the training process
    * Evaluation: If you want to solve a classificatio, accuracy is a good method; otherwise for regression R^2 is a good method
    * since its and iterative algorithm we specify the number of iterations 
    * r = model.fit(X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=100)
    * plt.plot(r.history['loss'], label = 'loss')
    * plt.plot(r.history['val_loss'], label='val_loss')
    
 ### Image as input to NN
 Dataset of images = No. of sample x Feature Dimension = No. of sample x (Height x Weight x Color)
 
* 1-Instantiate the model 
     model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28,28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10, activation='softmax')
     ])
     
    * dropout is a type of regularization in NN: 
       * it randomly drops input nodes by setting them to 0 so they have no influence on next layer
       * so the NN doesn't depend on a single input too strongly
       * there is a 20% chance of dropping a node in that layer (setting it to 0): ` tf.keras.layers.Dropout(0.2)`

* 2-Train the model 

    model.compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy']) 
                 
    model.fit(x_train, y_train)    
    
    * Cross entropy loss = -sum(y_k.log(y_k_hat)) where y_k_hat=p(y=k|x) and y=[0,0,...,1] (one hot encoding for each of k classes)
        * For correct prediction when y_k (one hot encoding) = 1 and y_k_hat = 1:
            * loss for k: -1*log1 = 0
        * For incorrect prediction when y_k (one hot encoding) = 1 and y_k_hat = 0:
            * loss for k: -1*log0 = inf
    
    * sparse_categorical_crossentropy  loss
        * its called sparse since the one-hot array is sparse (k-1 eateries are 0)
            * Cross entropy loss = -sum(y_k.log(y_k_hat))= -log(y_hat[k^*])
    
    
* 3-Evaluate the model and make prediction
    * model.evaluate(X,Y)
    * model.predict(X)       
     
     
### formulas
* cosine similarity cos(theta) = (a.b)/|a||b|
* pearson similarity
* filter: sliding pattern finder that passes through the image looking for a particular pattern
    
* `Valid Convolution`:Output length = N-K+1; if input length is N and kernel length is K output length is N-K+1
    * Given the input_image (A) and kernel(w)
    * output_height = input_height - kernel_height + 1
    * output_width = input_width - kernel_width + 1
        * deep learning version of the convolution is flipped (+) sign instead of (-) sign (its called cross correlation)
        * (i,j)th entry of the output = (A*w)_ij = sum_ii(sum_jj(A(i+ii,j+jj)w(ii,jj) for jj in[0:k-1]) for ii in [0:k-1])
        * X and Y are correlated means there is a degree of similarity between X and Y
        * convolution reverses the orientation of the filter whereas correlation does not
            * correlation neural network rather than convolution
* `Same Convolution`:Output length = N;  if we want the output to be the same size as the input (output_width=N) we use 0 padding around the input 
* `Full Convolution`:Output length = N+K-1; We could even extend the filter further and get non-zero outputs by `full-padding`
        * Input length = N
        * Kernel length = K
        * Output length = N+K-1
        
* Color image:
    * Input: HxWx3
    * Kernel: KxKx3
    * Output image: (H-K+1)x(W-K+1)

* Conv layer : sigmoid(W*x + b)
    * Convolution uses parameter sharing: using same weights in multiple places: color-image input=HxWx3 no_filters=64; kernel_size=(3, 3)
    * Input image: 32x32x3
    * Filter: 3x(5x5)x(64)
    * Output: 28x28x64 (32-5+1=28)
* Dense layer: sigmoid(W^T.x + b)

* Typical CNN architecture:
    * Step 1: feature transformer
        * CNN expects features of form X.shape : NxHxWxC and labels of length N
        * Conv > Pool > Conv > Pool > Conv > Pool > ...
        * Strided Conv > Strided Conv > Strided Conv > ...
    * Step 2:
        * Flatten()
        * GlobalMaxPooling2D()
    * Step 3: Non-linear classifier
        * Dense > Dense    
        * Activation and No. of nodes for the final layer depends on the task
        
    * when an image comes out of a convolution its 3d, H x W x C(No. of feature maps) but the Dense layers need a 1D input vector
        * we can use the Flatten() in keras to convert the 3D to 1D layer
        * or we can use the global max pooling
    * Global max pooling: we always get an output of 1X1XC (or just a vector of size c) regardless of H and W
        * Takes the max over each feature map
        * we don't care where the feature was found.
        * since the output is always a vector of size C,
          * max pooling allows the network to handle images of any size
        
    * Down-sampling: using pooling layer
        * If input is NxN, and pool size is M we have a N/M x N/M output (down-sample by M)
            * Convolution is a pattern finder (the highest number is the best matching location)
            * Max-pulling: return only the max in each square            
            * Average-pulling: return only the average in each square
        * Pool sizes are usually squared and small size compared to image size (3x3, 5x5, 7x7). 
            * The parameter that controls the box's overlap is stride. 
            * Stride tells us how far apart each sliding window should be. 
            * It tells you how many spaces the box should move (pull size of 2 and stride of 2 means no overlap)
            * We can get the same reduction in output size as pooling if we do stride convolution 
        * Increase No. of feature maps in each convolution (32>61>128>128)
        * Why we use pooling: 
            * If we down-sample (shrink) an image we have less data to process; 
            * but we also lose spacial information when we shrink the image,and we don't care where the feature was found
            * Translational invariance: we don't care where in the image the feature occurred, we just care it did

## Why Keras Functional API
* Easy to create branches
* Models with multiple inputs/outputs: e.g. model = Model(inputs=[i1,i2,i3], outputs=[o1,o2,o3])
    * Conv2D (e.g. image because there are 2 spacial dimensions; shape: HxW) 
    * Conv1D (e.g. speech only varies with time; shape: T) 
    * Conv3D (e.g. video; shape: HxWXT) (e.g.medical imaging data: Voxels; shape: HxWxD)
        * Pixel: (Picture Element)
        * Voxel: (Volume Element)
    * E.g. Conv2D(# of output feature maps:'32', Filter dimensions: '(3,3)', strides=2, activation function:'relu', padding:{default='valid','same','full'})(i)
* Example CNN if we have 2 spatial dimensions: 
    i = Input(shape=x_train[0].shape)
    x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
    x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dense (512, activation='relu')(x)
    x = Dense (512, activation='softmax')(x)
    model = Model(i,x)
    
   * You can use dropout in CNN but its not a good idea to suddenly remove have the pixels of the image 
    x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
    x = Dropout(0.2)(x)
    
## Generator and iterators
In Keras when we want to generate augmented data on the fly we use generators
* Generator with yield: no list is ever created, do not need to be stored in memory simultaneously
    def my_func():
        for _ in range(10):
            x = np.random.randn()
            yield(x)
            
    def data_augmentation_generator():
        for x_batch, y_batch in zip(x_train, y_train):
            x_batch = augment(x_batch)
            yield x_batch, y_batch       


    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizental_flip=True
        vertical_flip = False
        brightness_range = .. 
        shear_range = ..
        rotation_range = ..
        )
    
    train_generator = data_generator.flow(x_train, y_train, batch_size)
    
    steps_per_epoch = x_train.shape[0]
    r = model.fit_generator(
        train_generator, 
        steps_per_epoch=steps_per_epoch, 
        epochs=50)
    * fit_generator returns history and plot loss per itteration
        
    * steps_per_epoch = No. of samples/batc_size: 
        * so if we have 50 samples and batch_size is 10, then you have to itterate through 5 batches (5x10=50
        * steps_per_epoch: is how many steps it takes to see is the entire training set

## Time series
* any continuous valued measurement taken periodically (speech, stock price, airline passengers, weather tracking)
* shape of sequence : N x T x D
    * D = # features
    * N =  # samples
    * T = # time steps in the sequence (sequence length) 
    * e.g. GPS location data N(No. of people)xD(latitude,longitude)xT(No. of latitude, longitude measurements per unit of time)
    * e.g. given time by stock price measurements: for a (window of T=10)(price has m features if we measure n stocks D=m)(N=total number of windows in the time series)
        * for a seq of length L(100 stock prices for 1 stock) no. of windows of size T (e.g. length 10) = L-T+1 = 100-10+1 = 91
        * a single sample will be of size TxD=10x500
    * e.g. brain wave measurement from sensors on head; at each time step, each electrode measures voltage       
        * N = # of letters a person is typing
        * T=1000, 1 sec of measurements; sampling rate of 1 sample/millisecond
        * D= # of electrodes in the brain        
    * why NxTxD (constant length sequences)
        * tabular data: NxD
            * linear regression expects NxD
            * if the data is NxTxD (for 1-D time series, D=1, we pretend that T is D)
        * images : NxHxWxC
    
    * Auto Regressive(AR) model for estimating time series:(x_t)_hat = w_0 + w_1.x_(t-1) + w_2.x_(t-2) + w_3.x_(t-3) 
        * we want to predict multiple steps ahead (e.g. predict 3 days ahead from past 3 days) 
            * (x_4)_hat = w_0 + w_1.x_1 + w_2.x_2 + w_3.x_3 
            * (x_5)_hat = w_0 + w_1.x_2 + w_2.x_3 + w_3.(x_4)_hat  
            * (x_6)_hat = w_0 + w_1.x_3 + w_2.(x_4)_hat + w_3.(x_5)_hat  
    
* In DNNs every input feature is connected to the next hidden layer and not easy to exploit the model architecture
 However, in CNNs and RNNs we take advantage of the structure using shared weights.
    * we apply the same weight W_xh to each vector x_t and same W_hh to get from h_(t-1) to h_t
    * For DNN and CNN, the output is probability of a category given the input p(y=k|x)
    * For RNNs (y_T=k|x_1,x_2,...,x_T)
        * or p(x_(t+1)|x_t,x_(t-1),x_(t-2),....,x_2,x_1)
        * in RNNs most often we use 1 hidden layer (not 100 like CNN):
        * hidden state is the non-linear function of the past values and prediction is the linear function of past values 
    * For Markov Models, they make the Markov assumption that the current value depends only on immediate previous value
        * p(x_t|x_t-1, x_t-2,...,x_2,x_1) = p(x_t|x_t-1)
    

* RNNs
    * p(x_(t+1)|x_t,x_(t-1),x_(t-2),....,x_2,x_1)       
    * hidden vector loops back to itself: it depends not only on the input but also its own previous value
    * Given and input of size T where each input is of length T
        * input X: It is a TxD input matrix: x_1,x_2,..._x_T and shape(x_t)=D
        * w_xh: input to hidden weight
        * w_hh: hidden to hidden weight
        * b_h: hidden bias
        * W_o: hidden to output weight
        * b_o: output bias
        * tanh hidden activation and Softmax output activation
        * y_hat = []
        * h_last = h0
        * for t in in range(T):
            * h_t = tanh(X[t].dot(Wx)) + h_last.dot(Wh) + bh)
            * y_hat = softmax(h_t.dot(Wo) + bo)
            * h_last = h_t        
    * h(t) = sigmoid(W_(xh)^Transpose . x_t + W_(hh)^Transpose . h_(t-1) + b_h)
        * given x_1 we assume h_0 is an array of 0s in Tensorflow to estimate h(1) then we estimate (y_1)_hat
        * the we use x_2 and h_1 we estimate (y_2)_hat
        * we only keep (y_T)_hat and simply discard previous outputs
        * but in some cases we keep y_hat of previous steps (e.g. language translation where both input and outputs are sequences)
    * (y_t)_hat = sigmoid(W_o^Transpose = b_o)
    * There is a time delay of 1 step
* vanishing gradient: RNNs can't remember long term dependencies. the more deeply nested the more multiplication in the derivative(chain rule)
    * SimpleRNNs have no choice but to eventually forget due to vanishing gradient
    * LSTMs and GRUs are the solution
    * GRU: the hidden state becomes weighted sum of previous hidden states and new values. 
        * output is h(t) depends on h(t-1) and x(t)
        * it can decide to remember the previous state and allowing that to move forward in time or forgetting it
        * we use binary classifiers (logistic regression neurons) as our gates to remember/forget each component h(t-1)
        * same API as RNN.
        * GRU has less parameters than LSTMs. ButLSTM outperform the GRU most of the time
    * LSTMS: LSTMS return 2 states : hidden state h(t) and cell state c(t) (usually ignored)
        * output is h(t) and c(t) depends on h(t-1) and x(t) and c(t-1) and h_0 and c_0
        * NOT same API as RNN.
        * we have cell state (c(t)), which is simple weighted sum of previous cell state and simple RNN weighted by forget gate and input gate.
        * we have hidden state (h(t)), which is squashed version of the cell state with output gate controlling which values are allowed to pass through
        * forget gate, input/update gate, and output gate 
            * f(t): neuron binary classifier
            * i(t): neuron binary classifier            
            * o(t): neuron binary classifier 
            * c(t) = f(t) * c(t-1) + i(t) * SimpleRNN
            * h(t) = o(t) * tanh(c(t))
    
    SimpleRNN:
    i= Input(shape(T,D))
    x=SimpleRNN(M)(i)
    x=Dense(K)(x)
    model = Model(i,x)
    
    * LSTM 
    i =Input(shape(T,D))
    x=LSTM(M)(i)
    x=Dense(K)(x) # NxTxM
    model = Model(i,x)
    
    * LSTM, vs GRU, v Simple RNN
    o,h = simpleRNN(M, return_state=True)(i)
    o,h = GRU(M, return_state=True)(i)    
    o,h,c = LSTM(M, return_state=True)(i)
    


### Ways to forecast
* one-step forecasting: it artificially gives us good results
    * classical models likeArima build a 1 step predictor, the itteratively apply it to predit multiple steps
* multi-step forecasting: iteratively builds a multi-step forecast using
    * model's own prediction can lead to poor results, even on simple  problems like sine wave
    * e.g. a NN with multiple outputs: a Dense(units=12) can  predict 12 steps ahead
* Naive forecast: just predict the last value
    * stock prices closely follow a random walk so a naive forecast is the best
* Implementation: in TF wew have constant length time series so our data should be of form NxTxD array

### RNN for images
* Load in the data X of shape N x T x D (T=28,D=28)
* Instantiate the model LSTM -> Dense(10,activation = 'softmax'): final dense layer with 10 outputs and softmax activation
*  fit/plot the loss/accuracy etc

### Deep Recommendation system
Dataset: User - Item - Rating (incomplete)
Embedding: maps the category to feature vector 
### Rnn for text 
* tokenizer converts strings containing multiple words into a list of token and then convert those into a list of integers
* each integer corresponds to a unique word, which csan be used to index a weight matrix called an embedding
* to represent words we use word embedding generated by converting word to integers 
* we pad the sequences (of integers) to get a 2D
* one hot encoding of vocabulary is not a good solution as it does not convey the geometric relationship between words and requires too much space
    * one hot encoding an integer k annd multiplying it by a matrix is the same as selecting the k'th row of the matrix
        * one_hot(k) * W == W[k]
        * tensorflow embedding (TxD matrix): convert words into integers, use integers to index word embedding matrix to get word vectors for each word 
* So when we pass a NxT matrix of word indices and after passing an embeding layer we get an NxTxD tensor. 
* T = Sequence length, D = Embedding dimensionality , M = hidden vector dimensionality , K = Number of output classes
    * i = Input(shape=(T,))
    * x = Embedding(V,D)(i) # x is now NxTxD
    * ....the rest of RNN ... (dense layer does your classification or regression)
    * x = LSTM(M)(x)
      x = Dense (K, activation='softmax')(x)
    * x = LSTM(M,  return_sequences = True)(x)
      x = GlobalMaxPooling1D()(x)
      x = Dense(K, activation = 'softmax')(x) 
* Assign each word a D dimensional vector
* Tensorflow uses constant sequence length series so all our data fits into NxTxD array
    * T = max seq. length (any shorted sentences require 0 padding, so 0 can't be used as a word index)

### Transfer learning
The features we learnt from one task can be useful for another task. 
e.g. use pre-trained models for ImageNet for other tasks
* 2 approaches for transfer learning
    * use data augmentation with Image Data Generator
        * entire CNN computation must be inside loop
    * pre-compute Z without data augmentation
        * only need to train a logistic regression on (Z,Y)
        * freeze the weights in the body and train the weights in the head of network
* Examples:
    * VGG (Visual Geometry Group): A series of conv layers and pooling followed by a dense layer
    * Resnet: it is a CNN with branches
    * Inception: multiple con imn parallel
    * MobileNet: makes tradeoff 
    
### GANs: Generative Adversarial Network
* The main used case for GAN is to generate data (e.g. images)
* GANs are a system of 2 NNs each of them has its own objective
    * Generator: generate images
    * Discriminator: discriminate between real and fake images
* GANs are generative because they generate data and they are adversarial because
    you have 2 NNs which oppose eachother in an adversarial manner. They have opposite goals
    * The goal of the first NN is to generate images that look real
    * The goal of the second NN is to identify real and fake images

* The discriminator loss function is binary cross entropy for binary classifier  (to identify fake and real pic)
* The generator loss
    * freeze the discriminator layers so only generator is trained 
### Terminology

Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

* one epoch = one forward pass and one backward pass of all the training examples
* batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
* number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

*Advantages of using a batch size < number of all samples:
    * It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory. That's especially important if you are not able to fit the whole dataset in your machine's memory.
    * Typically networks train faster with mini-batches. That's because we update the weights after each propagation. In our example we've propagated 11 batches (10 of them had 100 samples and 1 had 50 samples) and after each of them we've updated our network's parameters. If we used all samples during propagation we would make only 1 update for the network's parameter.
* Stochastic is just a mini-batch with batch_size equal to 1. 
    * In that case, the gradient changes its direction even more often than a mini-batch gradient.
* Disadvantages of using a batch size < number of all samples:
    * The smaller the batch the less accurate the estimate of the gradient will be. The direction of the mini-batch gradient fluctuates much more in comparison to the direction of the full batch gradient.

###Loss/error/cost:
    * MSE: (1/N)*sum(y-y_hat)^2 :linear regression means using mse with a linear model
        * in Regression: we use MSE, data has "Gaussian" distribution, and error function is negative log likelihood and our solution is called maximum likelihood solution
        * if our continuous data has a Gaussian distribution the pdf p(x) is exponential  
        * likelihood L = multiplication of p(x_i) for all x_i values
        * log likelihood: set logL = 0 to derive mu and you will see mu=mean(x_i)
            * by maximizing L with respect to mu (we want L to be big when its close to most of the data where the mean of our data is mu)
        * instead of dL/dmu we first take the derivative of the log (maximize the log likelihood)
        * if you look at the formula you see maximizing the log likelihood is the same is minimizing the mse 
        
    * mae: (1/N)*sum(|y-y_hat|) if we use mean absolute error we dont call it linear regression anymore
    
    * binary cross entropy: it is the right loss function to use when you are doing binary classification
        * in Binary classification: we use binary cross entropy, data has "Bernoulli" distribution, and error function is negative log likelihood, and our solution is called maximum likelihood solution
        * the bernoulli distribution is used for  binary events
        * we use PMF for discrete rand variables, we use PDF for continuous random variable
        * pmf : p(x) = mu^x * (1 - mu) ^ (1-x)
        * Likelihood L =  multiplication of p(x_i) for all x_i values
        * log likelihood: set logL = 0 to derive mu and you will see mu=mean(x_i)
        * Loss function L = -1/N * sum( y_i * log y_i_hat  + (1 - y_i)log(1 - y_i_hat) )
        * we always devide by N because we want our loss to be invariant to the number of samples N
    
    * categorical cross entropy: it is the negative log likelihood for "categorical" distribution
        * in multi-class classification: we use categorical cross entropy, data has "categorical" distribution
            * The "regural" categorical cross-entropy uses the full NxK target: requires NxK multiplocatiom/addition
            * The "Sparse" categorical cross-entropy uses the original target (1-D array): requires N multiplications/addition
        * categorical PMF : p(x) = mult(mu_k)^1(x=k) for k in [1:K]
        * Likelihood : mult(p(x_i)) for i in [1:N]
        * log likelihood = categorical cross entropy = logL = sum(sum(1(x_i=k)log mu_k)for k in [1:K])for i in [1:N]) 
        * categorical cross entropy loss function : sum( y_i_k * log y_i_k_hat ) for k in [1:K] and for i in [1:N]
        * i = sample index, K= class label, y_ik = one-hot encoded value
    
    * Gradient descent (GD): to minimize the loss function with respect to model parameters we use Gradient and set it to 0 to solve and find the parameters (e.g. w, b)
        * some equations are not analaticaly solvable, so we use numerical approxomation
        * in GD, we repeatedly take small steps in the direction of the gradient to update paramet (e.g. w)
            * at each step loss function L(w) decreases provided that the step size  is small enough, and eventually we will converge to the minimum
            * one epoch means one pass of the full training set and we must choose the numberof epochs high enough so that the loss converges
            * Usually each epoch may contain a few iterations. Because usually we divide the training set into batches, each epoch goes through the whole training set.
            * step size (or learning rate) must be small enough so that the loss does not blow up, and large enough so we don't have to wait too long
            * the cost of gradient descent for each iteration will be very high that why we use SGD
    * Stochastic gradient descent (SGD): The word 'stochastic' means a system or a process that is linked with a random probability. Hence, in Stochastic Gradient Descent, 
    a few samples are selected randomly instead of the whole data set for each iteration. 
        * we split our data into batchesd (like 32,64,128) for each batch we compute the GD
    * GD with momentum: (0.9 is a good momentum) 
       * Gradient descent with momentum involves applying exponential smoothing to the computed gradient. This will speed up training, because the algorithm will oscillate less towards the minimum and it will take more steps towards the minimum.
       * Momentum is an added term in the objective function, which is a value between 0 and 1 that increases 
        the size of the steps taken towards the minimum by trying to jump from a local minimum. If the momentum term is large then the learning rate 
        should be kept smaller. A large value of momentum also means that the convergence will happen fast. But if both the 
        momentum and learning rate are kept at large values, then you might skip the minimum with a huge step. A small value of momentum cannot reliably 
        avoid local minima, and can also slow down the training of the system. Momentum also helps in smoothing out the variations, if the gradient keeps changing direction. 
    
    * variable and adaptive learning: decrease the learning rate as a function of time
        * step decay: we reduce the learning rate every N steps 
        * exponential decay: we reduce the learning rate exponentially
        * variable decay: we reduce the learning rate but dropoffs is slower than exponential decay 
        * AdaGrad: adaptive learning rate: steep gradient in one direction, flat another. 
            * Adapt learning rate for each parameter individually, based on how much learning it has done so far
                * each parameter of NN has its own cache. if a parameter had large gradients in the past its cache is large (small) and its effective learning rate is small (large) 
                * cache: decay rate * cache + (1-decay rate) * gradient ^2
        * RMSProp: AdaGrad decreases learning rate too aggressively since its cache is growing toofast, so lets decrease it on each update                
    
    * Adam optimization algorithm: stands for adaptive moment estimation. Briefly, this method combines momentum and RMSprop (root mean squared prop).

### Dimensionality reduction
SVD wants to preserve the structure of the data while reducing the dimensionality to increase the processing time
* z=V^T*x
* x: one sample is a vector x of length D
* output z: a transformed vector z of length d << D
* V is the Dxd matrix
* SVD: is about how to find V
*  Implementation:
    model = TruncateSVD()
    model.fit(X)
    z = model.transform(X)
* Theory
    * X = USV^T
    * z = V^T * x
        * X: (NxD), U: (NxD), S: (DxD), V^T : (DxD)
        * after dimension reduction X: (NxD), U: (Nxd), S: (dxd), V^T : (dxD)
        * S is a diagonal matrix: only  diagonal elements are not zero (it tells us how important each corresponding element of z is)
        * S^2 is covariance of Z, each element on diagonal of s is standard deviation of correspinding latent dimension z
        * S^2 and V contain eigenvalues and eigenvectors of the covariance of X
* In NLP given the term by doc matrix: NxD you can use
    * linear dimension reduction: ICA SVD PCA 
    * non-linear dimension reduction: t-SNE
    * impl;ementation: model = Model(); Z = model.fit_transform(X)