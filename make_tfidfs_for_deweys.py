## Inspirert av https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9

import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize

stop = set(stopwords.words('norwegian'))

def split_corpus_by_deweys(original_name):
    articles=open(original_name+'.txt',"r")
    articles=articles.readlines()
    dewey_dict={}
    training_set=[]
    test_set=[]
    antall_dewey_stor_nok = 0
    dewey_freq = {}

    for article in articles:
        dewey=article.partition(' ')[0].replace("__label__","")
        article_wo_stopwords_tokenized = [i for i in word_tokenize(article.lower()) if i not in stop]
        article_wo_stopwords = ' '.join(article_wo_stopwords_tokenized)
        if dewey in dewey_dict:
            dewey_dict[dewey].append(article_wo_stopwords)
        else:
            dewey_dict[dewey]=[article_wo_stopwords]
    #print("antall deweys:"+str(len(dewey_dict)))

    for key in dewey_dict.keys():
        temp = dewey_dict[key]

        # text_file = open("word2vec_per_dewey/"+str(key)+".txt","w")
        # for text in temp:
        #     text_file.write(text)
        # text_file.close()
        if len(temp) > 1:
            try:
                tfv = TfidfVectorizer(
                    min_df =0.05, max_df = 0.90, max_features=20, strip_accents='unicode',
                    analyzer="word", ngram_range=(1,1),
                    use_idf=1, smooth_idf=1, sublinear_tf=1, token_pattern= '\\w{5,}')

                tfidf_matrix = tfv.fit_transform(temp)
                scores = zip(tfv.get_feature_names(),
                             np.asarray(tfidf_matrix.sum(axis=0)).ravel())
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)


            except ValueError:
                print("Her skjedde noe feil. på dewey:"  + str(key))
                pass
            tfidf_file = open("tfidfs/" + str(key) + "tf-idf", "w")
            tfidf_tuples = []
            for item in sorted_scores:
                tfidf_tuples.append((item[0],item[1]))
                #print("{0:50} Score: {1}".format(item[0], item[1]))
                tfidf_file.write("{0} {1}".format(item[0], item[1]) + '\n')
            #print(str(item[0])+":" +str(item[1]))
            tfidf_file.close()
            #print(len(tfidf_tuples))
            #print("LOLOLO:"+str(key))
            #print(tfidf_matrix)
            #print(tfidf_tuples[10])



if __name__ == '__main__':
    split_corpus_by_deweys("combined3deweys")