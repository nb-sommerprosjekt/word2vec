import gensim
import nltk
import os
import pickle
from nltk.corpus import stopwords

def make_word2vec_models_for_dewey():
    file_name = []
    full_file_name = []
    for file in os.listdir("corpus_separated_deweys/"):
        if file.endswith(".txt"):
            full_file_name.append(file)
            file_name.append(file.replace(".txt",""))
            os.mkdir("models/"+file.replace(".txt",""))
    print(file_name)
    print(full_file_name)

    #Making vocabulary
    with open("full_text_non_stemmed.txt") as f:
            vocab_sentences = f.readlines()
    vocab_sentences_tokenized =  []
    for sent in vocab_sentences:
       vocab_sentences_tokenized += [nltk.word_tokenize(sent)]

    for text_file in full_file_name:
        with open("corpus_separated_deweys/"+text_file) as f:
            sentences = f.readlines()
        sentences_tokenized = []
        for sent in sentences:
            sentences_tokenized += [nltk.word_tokenize(sent)]


        model = gensim.models.Word2Vec(iter=1)
        model.build_vocab(vocab_sentences_tokenized)
        model.train(sentences_tokenized, total_examples= model.corpus_count, epochs= 30)
        filename = text_file.replace(".txt","")
        model.save("models/"+filename+"/"+filename)
        print("Ferdig!!!")

def make_full_model():
    with open("full_text_non_stemmed.txt") as f:
            vocab_sentences = f.readlines()
    vocab_sentences_tokenized =  []
    for sent in vocab_sentences:
       vocab_sentences_tokenized += [nltk.word_tokenize(sent)]

    model = gensim.models.Word2Vec(iter=1)
    model.build_vocab(vocab_sentences_tokenized)
    model.train(vocab_sentences_tokenized, total_examples= model.corpus_count, epochs= 30)
    model.save("full.bin")
    print("Ferdig!!!")
def make_list_of_similar_words(deweynr,model_path):
    with open("klassebetegnelser_dict.pckl","rb") as f:
        klassebetegnelser = pickle.load(f)
    meta_dewey = open("meta/"+deweynr+"-meta.txt", "w")
    klassebetegnelse_wo_stopwords = []
    klassebetegnelse_tokenized = []
    #for words in klassebetegnelser[deweynr]:
    klassebetegnelse_tokenized = nltk.word_tokenize(klassebetegnelser[deweynr].lower())

    klassebetegnelse_wo_stopwords = [word for word in klassebetegnelse_tokenized if word not in stopwords.words('norwegian') ]


    model = gensim.models.Word2Vec.load(model_path+'/'+deweynr+"/"+deweynr)
    similars = []
    meta_dewey.write(klassebetegnelser[deweynr]+'\n')
    for words in klassebetegnelse_wo_stopwords:
        print(words)
        try:
             meta_dewey.write(words+":::"+str(model.wv.similar_by_word(words, topn=5))+"\n")
            #similars.append(model.wv.similar_by_word(words, topn=5))
        except KeyError:
              meta_dewey.write(words+":::"+"not in vocabulary"+"\n")
              #similars.append("word " +words + "var ikke i vokabularet")
              print(words +" : var ikke i vokabular")
              pass
    #similars = gensim.models.KeyedVectors.similar_by_word(word = klassebetegnelser[deweynr],topn = 5)
    print(klassebetegnelser[deweynr])
    print(similars)
    #print(type(similars))
    meta_dewey.close()

if __name__ == '__main__':
    #make_full_model()
    #make_word2vec_models_for_dewey()
    # for file in os.listdir("corpus_separated_deweys/"):
    #     if file.endswith(".txt"):
    #         #print(file.replace(".txt",""))
    #         make_list_of_similar_words(file.replace(".txt", ""), "models")




