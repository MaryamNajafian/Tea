from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Just a simple test
sentences = [
    "I like eggs and ham.",
    "I love chocolate and bunnies.",
    "I hate onions."
]

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE) # instantiate the model
tokenizer.fit_on_texts(sentences) # fit the model
sequences = tokenizer.texts_to_sequences(sentences) # transform the data

print(sequences) # [[1, 3, 4, 2, 5], [1, 6, 7, 2, 8], [1, 9, 10]]

# word to index mapping ( index 0 is reserved for 0 padding)
print(tokenizer.word_index) # {'i': 1, 'and': 2, 'like': 3, 'eggs': 4, 'ham': 5, 'love': 6, 'chocolate': 7, 'bunnies': 8, 'hate': 9, 'onions': 10}

# use the defaults: max sequence length = max sentence length
data = pad_sequences(sequences)
print(data)
# [[ 1  3  4  2  5]
#  [ 1  6  7  2  8]
#  [ 0  0  1  9 10]]

# pre-padding
MAX_SEQUENCE_LENGTH = 5
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(data)

# post-padding
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print(data)

# too much padding
data = pad_sequences(sequences, maxlen=6)
print(data)

# truncation to cut off beginning of the sentences
# because an RNN pays more attention to the end of sequences
data = pad_sequences(sequences, maxlen=4)
print(data)

# truncation to cut off ending of the sentences
data = pad_sequences(sequences, maxlen=4, truncating='post')
print(data)