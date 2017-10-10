### DENNE KODEN VIRKER IKKE HELT ENDA. SKAL FIKSES ETTERHVERT. INSPIRERT av MLP fra http://nadbordrozd.github.io/blog/2017/08/12/looking-for-the-text-top-model/



from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation
import numpy as np
import gensim
from keras.layers import Embedding
from keras.models import Model, Sequential
import pydot
import graphviz
def get_articles(original_name):
    # Tar inn en textfil som er labelet på fasttext-format. Gir ut to arrays. Et med deweys og et med tekstene. [deweys],[texts]
    articles=open(original_name+'.txt',"r")
    articles=articles.readlines()
    dewey_array = []
    docs = []
    dewey_dict = {}
    for article in articles:
        dewey=article.partition(' ')[0].replace("__label__","")
        article_label_removed = article.replace("__label__"+dewey,"")
        docs.append(article_label_removed)
        dewey_array.append(dewey)

    return dewey_array, docs
dewey_train, text_train = get_articles("corpus_w_wiki/data_set_100/combined100_training")
# dewey_train, text_train = get_articles("full_text_non_stemmed")
#dewey_train, text_train = get_articles("full_training1000words_min100")
#dewey_test, text_test = get_articles("test_1000words_min100")
# print(len(dewey_test))
# print(len(dewey_train))
# print(len(set(dewey_test)))
# print(len(set(dewey_train)))



labels_index = {}
labels = []
for dewey in set(dewey_train):
    label_id = len(labels_index)
    labels_index[dewey] = label_id
for dewey in dewey_train:
    labels.append(labels_index[dewey])
print(len(labels_index))
print(labels_index)
print(len(labels))

#dewey_train = [int(i) for i in dewey_train ]
## Preprocessing
vocab_size =5000
MAX_SEQUENCE_LENGTH = 5000
num_classes = len(set(dewey_train))
tokenizer = Tokenizer(num_words= vocab_size)
tokenizer.fit_on_texts(text_train)
sequences = tokenizer.texts_to_sequences(text_train)
sequence_matrix = tokenizer.sequences_to_matrix(sequences, mode = 'binary')
print(len(sequences))
print(sequence_matrix.shape)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print(data.shape)
print(labels.shape)

# split the data into a training set and a validation set
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
print(labels.shape)
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data#[:-nb_validation_samples]
y_train = labels#[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]
#
# print(x_val[0])
# print(y_val[0])
print(x_train.shape)
print(y_train.shape)
#print(x_val.shape)
#print(y_val.shape)
model = Sequential()
model.add(Dense(128, input_shape=(MAX_SEQUENCE_LENGTH,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=64,
          epochs=40,
          verbose=1
          )

model.save("keras_models/keras_deep_wiki_MLP100.bin")

####Preparing test_set
dewey_test, text_test = get_articles("corpus_w_wiki/data_set_100/100_test")
test_labels = []

test_label_index = {}

for dewey in dewey_test:
    test_labels.append(labels_index[dewey])

test_sequences = tokenizer.texts_to_sequences(text_test)
test_sequence_matrix = tokenizer.sequences_to_matrix(test_sequences, mode = 'binary')
test_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(test_word_index))
test_data = pad_sequences(test_sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)
test_labels = to_categorical(np.asarray(test_labels))
x_val = test_data
y_val = test_labels



#
score = model.evaluate(x_val, y_val, batch_size= 64, verbose=1)
print('Test_score:', score[0])
print('Test Accuracy', score[1])
# # print(score)
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file = 'model.png')
# model.save("keras_deep_20epochs_MLP100.bin")