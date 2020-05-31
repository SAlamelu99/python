
#import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#load data
#LOADING DATA and opening theinput data in the form of txt file
#project  Gutenberg is where the data can be found 
file = open("textfile.txt").read()


#tokenization 
#standardization

def tokenize_words(input):
    #lowercase everything
    input = input.lower()
    #instantiating the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    #tokenizing the text into tokens
    tokens = tokenizer.tokenize(input)
    #filtering the stopwords using lambda
    filtered = filter(lambda token:token not in stopwords.words('english'),tokens)
    return " ".join(filtered)
#preprocessing the input data
processed_inputs = tokenize_words(file)
    

#chars to numbers
#convert characters in our input to numbers
#sorting the list of the set of all characters that appear in out i/p text and then use the enumerate function to get no that represent characters
#then create a dictionary that stores the keys and values, or the characters and the nmumbbers that represent thm
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i, c in enumerate(chars))

#check if words to chars or chars to num has worked
#printing the length of our variables
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:", input_len)
print("Total vocab:",vocab_len)


#seq length
#defining how long we want an individual sequence
#individual sequence is a complete mapping of input characters as integers
seq_length = 100
x_data = []
y_data = []

#loop through the sequence
#going through the entire list of inputs and conv characters to numbers with for loop
#creating a bunch of sequence where each sequence starts with the next character in the input data
#beginning with the first character
for i in range(0, input_len - seq_length,1):
    #define input and output sequence
    #input is  the current character and the desired sequence length
    in_seq = processed_inputs[i:i + seq_length]
    #out sequence is the initial character and total sequence length
    out_seq = processed_inputs[i + seq_length]
    #converting the list of charaters to integers based on previous values and appending the values to our list
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
#check to see total sequences we have
n_patterns = len(x_data)
print("Total Pattern:",n_patterns)


# convert input sequence to np array that our network can use 
x = numpy.reshape(x_data,(n_patterns,seq_length,1))
x = x/float(vocab_len)

#one hot encoding our label data
y = np_utils.to_categorical(y_data)

# creating the model
#creating the sequewntial model
#dropout is used to prevent overfitting
model = Sequential()
model.add(LSTM(256,input_shape = (x.shape[1],x.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))


#compile the model
model.compile(loss = 'categorical_crossentropy',optimizer='adam')

#saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose = 1,save_best_only = True,mode='min')
desired_callbacks = [checkpoint]


#fit model and let it train
model.fit(x,y,epochs=4, batch_size = 256,callbacks=desired_callbacks)


#recompile models with the saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')


#output of the model back into the character
num_to_char = dict((i,c) for i, c in enumerate(chars))


#random seed to help generate
start = numpy.random.randint(0,len(x_data)-1)
pattern = x_data[start]
print("Random Seed:")
print("\"",''.join([num_to_char[value] for value in pattern]),"\"")


#generate the text
for i in range(1000):
    x = numpy.reshape(pattern,(1,len(pattern),1))
    x=x/float(vocab_len)
    prediction = model.predict(x,verbose=0)
    index=numpy.argmax(prediction)
    result=num_to_char[index]
    seq_in=[num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:]



