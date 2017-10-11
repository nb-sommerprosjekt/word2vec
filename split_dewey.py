import os
import re
import math
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

#This function takes text file(originalname) and creates a test.txt set and training set
# with a split based on test_percentage and minimum number of articles needed per deweynumber.
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
        test_text = open("corpus_separated_deweys/test.txt/"+key+".txt","w")
        train_text = open("corpus_separated_deweys/train/"+key+".txt","w")
        for text in temp:
             text=text.replace('__label__'+key,'')
        #     dewey_text_file.write(text)
        temp_test = str
        temp_train = str
        dewey_freq[key] = len(temp)
        if len(temp)>=min_art:
            antall_dewey_stor_nok=antall_dewey_stor_nok+1
            if len(temp)>1:
                split=max(1,math.floor(len(temp)*test_percentage))
                test_set.extend(temp[:split])
                temp_test = " ".join(temp[:split])
                test_text.write(temp_test)
                training_set.extend(temp[split:])
                temp_train = " ".join(temp[split:])
                train_text.write(temp_train)
            else:
                training_set.extend(temp)
        test_text.close()
        train_text.close()
    shuffle(training_set)
    print("antall deweys 100: " +str(antall_dewey_stor_nok))
    print(dewey_freq)
    # dewey_list = sorted(dewey_freq.items())
    # deweys,freq = zip(*dewey_list)
    # x = np.array(deweys)
    # y = np.array(freq)
    # #plt.bar(deweys, freq, width = 1.0, color = 'g')
    # plt.plot(x,y)
    # plt.show()
    shuffle(test_set)
    training="".join(training_set)
    test= "".join(test_set)
    f=open(training_name+".txt","w")
    f.write(training)
    f = open(test_name + ".txt", "w")
    f.write(test)
def split_text_by_dewey(text):

    articles=open(text+'.txt',"r")
    articles=articles.readlines()

    for text in temp:
         text=text.replace('__label__'+key,'')
         dewey_text_file.write(text)
    dewey_text_file.close()

#create_test_and_training_set("training_meta2_filtered", "test_meta2_filtered","training_meta2_filtered",0.20,2)
split_text_by_dewey()