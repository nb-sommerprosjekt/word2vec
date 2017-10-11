from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
import gensim
from keras.layers import Embedding
from keras.models import Model
import datetime
import time
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
def cnn(training_set, test_set, VOCAB_SIZE, MAX_SEQUENCE_LENGTH,EPOCHS, folder_to_save_model, loss_model):
    start_time = time.time()
    dewey_train, text_train = get_articles(training_set)
    dewey_test, text_test = get_articles(test_set)


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
    vocab_size = VOCAB_SIZE
    #MAX_SEQUENCE_LENGTH = 1000
    num_classes = len(set(dewey_train))

    tokenizer = Tokenizer(num_words= vocab_size)
    tokenizer.fit_on_texts(text_train)
    sequences = tokenizer.texts_to_sequences(text_train)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print(data.shape)
    print(labels.shape)

    # split the data into a training set and a validation set
    EMBEDDING_DIM = 300
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    print(labels.shape)



    test_labels = []
    for dewey in dewey_test:
        test_labels.append(labels_index[dewey])
    test_sequences = tokenizer.texts_to_sequences(text_test)

    test_word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(test_word_index))

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    test_labels = to_categorical(test_labels)



    x_train = data
    y_train = labels
    x_val = test_data
    y_val = test_labels



    w2v_model = gensim.models.Doc2Vec.load("doc2vec_dir/100epoch/doc2vec_100.model")


    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    j=0
    k=0
    for word, i in word_index.items():
        #embedding_vector = w2v_model[word]
        k = k+1
        try:
            if w2v_model[word] is not None:
            # words not found in embedding index will be all-zeros.

                embedding_matrix[i] = w2v_model[word]
        except KeyError:
            j=j+1
            continue

    sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype = 'int32')
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation = 'relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation = 'relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128,5, activation = 'relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)

    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss=loss_model,
                  optimizer="rmsprop",
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=EPOCHS
              )

    score = model.evaluate(x_val, y_val, batch_size= 64, verbose=1)
    print('Test_score:', score[0])
    print('Test Accuracy', score[1])
    time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

    save_model_path= folder_to_save_model+"cnn-"+str(vocab_size)+"-"+str(MAX_SEQUENCE_LENGTH)+"-"+str(EPOCHS)+"-"+str(time_stamp)

    time_elapsed = time.time() - start_time

    result_file = open("corpus_w_wiki/result_cnn","a")
    result_file.write("Time:"+str(time_stamp)+'\n'
                      + "Time elapsed: " + str(time_elapsed) + '\n'
                      +"Training set:"+training_set+"\n"+
                      "test_set:"+test_set+'\n'
                      +"Vocab_size:"+str(vocab_size)+'\n'
                      +"Max_sequence_length:"+str(MAX_SEQUENCE_LENGTH)+'\n'
                      +"test_score:"+str(score[0])+'\n'
                      +"test_accuracy:"+str(score[1])+'\n'
                      +"Epochs:" + str(EPOCHS)+'\n'
                      +"Antall deweys:" + str(num_classes)+'\n'
                      +"Antall docs i treningssett:" +str(len(dewey_train))+'\n'
                      +"Antall docs i testsett:" + str(len(dewey_test))+'\n'
                      +"saved_model:" + save_model_path +'\n'
                      +"loss_model:" + str(loss_model) +'\n'+'\n'
                      )
    result_file.close()
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file = 'model.png')
    model.save(save_model_path)
# Y_preds = model.predict(dewey_test)
# print(y_preds)

if __name__ == '__main__':
    try:
        cnn("corpus_w_wiki/data_set_100/combined100_training",
            "corpus_w_wiki/data_set_100/100_test",
            VOCAB_SIZE=1000,
            MAX_SEQUENCE_LENGTH =1000,
            EPOCHS=1,
            folder_to_save_model="keras_models/",
            loss_model= "categorical_crossentropy",
            )
    except TypeError:
                print("Error logget")
                pass
    print("Prosessen fortsetter")