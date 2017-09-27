# Inspired by https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/py3_poc.py
import gensim
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(list(word2vec)[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
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
             word2vecDoc.append(model[word.lower()])
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
    model = gensim.models.Doc2Vec.load("doc2vec_dir/doc2vec.model")
    print("Model initialisert")
    with open("test.txt","r") as text_file:
        text = text_file.read().replace('\n','')
    #print(text)

    dewey_train, text_train = get_articles("training_meta100_filtered")
    w2v_train = []
    for text in text_train:
        w2v_train.append(make_word2vecs_from_doc(text))
    print("Treningssett gjort om til word2vec")

    dewey_train , text_test = get_articles("test_meta100_filtered")
    w2v_test = []
    for text in text_train:
        w2v_test.append(make_word2vecs_from_doc(text))
    print("test-sett gjort om til word2vec")



    #w2v = make_word2vecs_from_doc(text)
    model_pipe = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    print("modellen er produsert")
    model_pipe.fit(text_train[:1],dewey_train[:1])

    print(text_test[10])
    print(sorted(set(dewey_train[:100])))
    filename = 'finalized_model.sav'
    print(model_pipe.predict([text_test[10]]))
    print(model_pipe.predict_proba([text_test[10]]))
    joblib.dump(model_pipe, "Modell.pckl")
    #model_pipe.predict_proba(["Genetikk"])
    #print(len(word2vecDoc))