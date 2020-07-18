#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
#%%
# download the data
df = pd.read_csv('./spam.csv',encoding='ISO-8859-1')
print(df.head(2))
df=df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
print(df.head(2))
df.columns = ['labels','data']
print(df.head(2))

# create binary labels
df['b_labels'] = df['labels'].map({'ham':0,'spam':1})
Y = df['b_labels'].values

# split up the data (the input and output are still pandas object)
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)


#%%
# Convert sentences to sequences of integers
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)

# get word -> integer mapping
word2idx = tokenizer.word_index
V = len(word2idx)
print('Found %s unique Tokens.', V)

# pad sequences so that we get a N x T matrix
data_train = pad_sequences(sequences_train)
print('Shape of data train tensor:', data_train.shape) #(3733, 162)

# get sequence length
T = data_train.shape[1] # T=162

data_test = pad_sequences(sequences_test, maxlen=T)
print('Shape of data test tensor:', data_test.shape)
#%%
# Create the model
# for a 1-D convolution, we need a TxD input,
# which we get after an embedding of seq. of
# words with length T

# data shrink in time dimension and grows in feature dimension as we add layers
D = 20  # embedding dimension
i = Input(shape = (T,))
x = Embedding(V+1,D)(i) # output TxD :(vocabindex+1xembedding dimension)
x = Conv1D(filters=32, kernel_size=3, activation='relu')(x) #
x = MaxPooling1D(pool_size=3)(x)
x = Conv1D(filters=64, kernel_size=3,activation='relu')(x)
x = MaxPooling1D(pool_size=3)(x)
x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(units=1, activation='sigmoid')(x)
model = Model(i,x)
#%%
# Compile and fit
model.compile(
    loss = 'binary_crossentropy',
    optimizer= 'adam',
    metrics= 'accuracy'
)

print('Training model...')
r = model.fit(
  data_train,
  Ytrain,
  epochs=5,
  validation_data=(data_test, Ytest)
)

# Plot loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()