# Inspired by https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/py3_poc.py
import gensim
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from collections import defaultdict

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


if __name__ == '__main__':
    w2v_model = gensim.models.Doc2Vec.load("doc2vec_dir/100epoch/doc2vec_100.model")
    print("Model initialisert")
    with open("test.txt","r") as text_file:
        text = text_file.read().replace('\n','')
    #print(text)

    dewey_train, text_train = get_articles("training_min100")
    w2v_train = []
    for text in text_train:
        w2v_train.append(make_word2vecs_from_doc(text))
    print("Treningssett gjort om til word2vec")

    dewey_test , text_test = get_articles("test_min100")
    w2v_test = []
    for text in text_test:
        w2v_test.append(make_word2vecs_from_doc(text))
    print("test-sett gjort om til word2vec")



    #w2v = make_word2vecs_from_doc(text)
    etree_model_pipe = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v_model)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=1000))])

    SVC_model_pipe = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v_model)),
                          ("SVM", SVC())])
    print("modellen er produsert")
    etree_model_pipe.fit(text_train,dewey_train)
    SVC_model_pipe.fit(text_train,dewey_train)
    print("Modellen er trent. Predikering pågår.")
    print(text_test[10])
    print(sorted(set(dewey_train[:100])))
    etree_results = []
    SVC_results = []
    i = 0
    for article in text_test:
        etree_results.append(etree_model_pipe.predict([article]))
        SVC_results.append(SVC_model_pipe.predict([article]))
    i = 0
    riktig = 0
    for result in etree_results:
        if result == dewey_test[i]:
            riktig = riktig +1
        i = i +1
    SVC_riktig = 0
    j = 0
    for res in SVC_results:
        if res == dewey_test[i]:
            SVC_riktig = SVC_riktig +1
        j = j +1
    print(len(etree_results))
    print(len(dewey_test))
    print(riktig)
    print ("Results Etree: "+str(riktig/len(dewey_test)))
    print ("Results SVC: "+str(SVC_riktig/len(dewey_test)))


    # print(model_pipe.predict([text_test[10]]))
    # print(model_pipe.predict_proba([text_test[10]]))


    #joblib.dump(model_pipe, "modell.sav", compress = 1)