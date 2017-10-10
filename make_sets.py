import os
import re
import math
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

def create_test_and_training_set(original_name, test_name,training_name,test_percentage,min_art):
    articles=open(original_name+'.txt',"r")
    articles=articles.readlines()
    dewey_dict={}
    training_set=[]
    test_set=[]
    antall_dewey_stor_nok = 0
    dewey_freq = {}

    for article in articles:
        dewey=article.partition(' ')[0].replace("__label__","")
        if dewey in dewey_dict:
            dewey_dict[dewey].append(article)
        else:
            dewey_dict[dewey]=[article]
    print("antall deweys:"+str(len(dewey_dict)))

    for key in dewey_dict.keys():
        temp = dewey_dict[key]
        shuffle(temp)
        dewey_freq[key] = len(temp)
        if len(temp)>=min_art:
            antall_dewey_stor_nok=antall_dewey_stor_nok+1
            if len(temp)>1:
                split=max(1,math.floor(len(temp)*test_percentage))
                test_set.extend(temp[:split])
                training_set.extend(temp[split:])
            else:
                training_set.extend(temp)

    shuffle(training_set)
    print("antall deweys 100: " +str(antall_dewey_stor_nok))
    print(dewey_freq)
    # dewey_list = sorted(dewey_freq.items())
    # deweys,freq = zip(*dewey_list)
    # newlist_dewey = [int(i) for i in deweys]
    # plt.plot(newlist_dewey,freq)
    # #plt.bar(newlist,dewey_freq.values(), width = 1.0, color='g')
    # plt.show()
    shuffle(test_set)
    training="".join(training_set)
    test= "".join(test_set)
    f=open(training_name+".txt","w")
    f.write(training)
    f = open(test_name + ".txt", "w")
    f.write(test)

create_test_and_training_set("corpus_w_wiki/only_wiki", "corpus_w_wiki/only_wiki100_test","corpus_w_wiki/only_wiki100_train",0.2,100)