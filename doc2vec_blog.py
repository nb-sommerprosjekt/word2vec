# Inspired by https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/py3_poc.py
import gensim
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from collections import defaultdict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        #self.dim = len(list(word2vec)[0])
        self.dim = 300
        #self.dim = len(list(word2vec.values())[0])
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    #or [np.zeros(300)], axis = 0)
                    or [np.zeros(self.dim)], axis = 0 )
            for words in X

        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        #self.dim = len(list(word2vec)[0])
        self.dim = 300
        self.word2weight = None
        #self.dim = len(list(word2vec.values())[0])
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)

        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda : max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    #or [np.zeros(300)], axis = 0)
                    or [np.zeros(self.dim)], axis = 0 )
            for words in X

        ])

def make_word2vecs_from_doc(text):
    #Tar en string, finner word2vec representasjon for hver av ordene, returnerer vector som inneholder
    # den opprinnelige stringen represenetert som word2vec for hvert ord. Altså [word2vec(ord1), word2vec(ord2)....]
    word2vecDoc = []
    for word in text.split():
        try:
             #print(model[word.lower()])
             word2vecDoc.append(w2v_model[word.lower()])
        except KeyError:
            pass
    return word2vecDoc

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

def print_results(testName,res_vector,dewey_test):

    return "Results "+testName+": "+str(accuracy_score(dewey_test,res_vector))


if __name__ == '__main__':
#    w2v_model = gensim.models.Doc2Vec.load("doc2vec_dir/100epoch/doc2vec_100.model")
    w2v_model = gensim.models.Doc2Vec.load("doc2vec_dir/doc2vec_w_wiki_30epoch/doc2vec_w_wiki_30epochs")
    print("Model initialisert")

    #dewey_train, text_train = get_articles("training_min500")
    #dewey_test , text_test = get_articles("test_min500")


    dewey_train, text_train = get_articles("corpus_w_wiki/Datasett_100_w_wiki/train_w_wiki100")
    dewey_test , text_test = get_articles("corpus_w_wiki/Datasett_100_w_wiki/test_min100")
    ### Test 0 Etrees
    etree_model_pipe = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_model)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=400))])
    print("Etree-modellen er produsert")
    etree_model_pipe.fit(text_train,dewey_train)
    print("E-tree Modellen er trent. Predikering pågår.")
    i = 0
    etree_results = []
    ### TEST 1 Etrees med tfidf
    etree_tfidf_model_pipe = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v_model)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=400))])
    print("Etree-modellen er produsert")
    etree_tfidf_model_pipe.fit(text_train,dewey_train)
    print("E-tree Modellen er trent. Predikering pågår.")
    i = 0
    etree_tfidf_results = []


    # TEST 2 SVC + embeddings
    SVC_model_pipe = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_model)),
                          ("SVM", SVC())])
    print("Etree-modellen er produsert")
    SVC_model_pipe.fit(text_train,dewey_train)
    print("SVC modellen er trent. Predikering pågår.")
    SVC_results = []


    ## TEST 3 SVM med TFIDF, uten embeddings

    SVM_tfidf= Pipeline([('tfidf_vectorizer', TfidfVectorizer(analyzer= lambda x: x)), ('linear_svc', SVC(kernel ="linear"))])
    SVM_tfidf.fit(text_train,dewey_train)
    SVM_tfidfresults = []
    print("Test 3 SVM w/tfidf - Done ")
    #Test 4 Multinomial Naive Bayes
    mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    mult_nb.fit(text_train,dewey_train)
    mult_nb_res = []
    print("Test 4 Multinomial Naive Bayes - Done")
    # Test 5 Bernoulli nb med count vectorizer
    bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    bern_nb.fit(text_train, dewey_train)
    bern_nb_res = []
    print("Test 5  Bernoulli nb med count vectorizer - Done")
    # Test 5 multinomial bayes med tfidf
    mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    mult_nb_tfidf.fit(text_train,dewey_train)
    mult_nb_tfidf_res = []
    print("Test 5 multinomial bayes med tfidf - Done")
    # Test 6 bernoulli naive bayes med tfidf
    bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    bern_nb_tfidf.fit(text_train,dewey_train)
    bern_nb_tfidf_res = []
    print("Test 6 bernoulli naive bayes med tfidf - Done")



    for article in text_test:
        etree_results.append(etree_model_pipe.predict([article]))
        etree_tfidf_results.append(etree_tfidf_model_pipe.predict([article]))
        SVC_results.append(SVC_model_pipe.predict([article]))
        SVM_tfidfresults.append(SVM_tfidf.predict([article]))
        mult_nb_res.append(mult_nb.predict([article]))
        bern_nb_res.append(bern_nb.predict([article]))
        mult_nb_tfidf_res.append(mult_nb_tfidf.predict([article]))
        bern_nb_tfidf_res.append(bern_nb_tfidf.predict([article]))
    #
    # res_dict={
    #     'etree' : accuracy_score(dewey_test,etree_results),
    #     'etree_tfidf' : accuracy_score(dewey_test,etree_tfidf_results),
    #     'SVC_embedding': accuracy_score(dewey_test,SVC_results),
    #     'SVM tfid' : accuracy_score(dewey_test,SVM_tfidfresults),
    #     'mult_nb' : accuracy_score(dewey_test,mult_nb_res),
    #     'bern_nb' : accuracy_score(dewey_test,bern_nb_res),
    #     'mult_nb_tfidf' : accuracy_score(dewey_test,mult_nb_tfidf_res),
    #     'bern_nb_tfidf' : accuracy_score(dewey_test,bern_nb_tfidf_res)
    # }

    # plt.bar(list(res_dict.keys()),list(res_dict.values()),width = 1.0, color ='g')
    # plt.show()
    print(print_results("Etree-embedding", etree_results, dewey_test))
    print(print_results("Etree-embedding w/tfidf",etree_tfidf_results,dewey_test))
    print(print_results("SVC_embedding",SVC_results,dewey_test))
    print(print_results("SVM_tfidf_test", SVM_tfidfresults, dewey_test))
    print(print_results("Multinomial naive bayes", mult_nb_res, dewey_test))
    print(print_results("Bernoulli Naive Bayes", bern_nb_res, dewey_test))
    print(print_results("Multinomial naive bayes w/tfidf", mult_nb_tfidf_res, dewey_test))
    print(print_results("Bernoulli Naive Bayes w/tfidf", bern_nb_tfidf_res, dewey_test))


