import os
import re
import math
from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle


def important_words(title):
    tittel = title.replace('\n', '')
    tokenized_tittel = word_tokenize(tittel)
    filtered_title = [word for word in tokenized_tittel if word not in stopwords.words('norwegian')]
    filtered_title2 = [word for word in filtered_title if not word.isdigit() and len(word) > 2]

    tittel_string_filtered = ' '.join(filtered_title2)
    return (tittel_string_filtered)


# Only used when creating the corpus for fasttext. Creates one large file with every line consisting of
# __label__"DEWEY" + text. Outputs the result in name.txt Input must be in the folder "folder".
def to_fasttext_keywords(folder, name):
    rootdir = folder
    label_file = open(name + '.txt', 'a')
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if str(file)[:5] == "meta-":
                f = open(os.path.join(subdir, file), "r+")
                meta_tekst = f.read()
                keyword = re.search('word:::(.+?)\n', meta_tekst)
                if keyword:
                    found = keyword.group(1)
                print(found.replace(' ', '-'))
                file_name = os.path.join(subdir, file)
                file_name_text = file_name.replace("meta-", "")
                tekst_fil = open(os.path.join(subdir, file_name_text), "r+")
                tekst = tekst_fil.read()

                label_file.write('__label__' + found + ' ' + tekst + '\n')


# Only used when creating the corpus for fasttext. Creates one large file with every line consisting of
# __label__"DEWEY" + text. Outputs the result in name.txt Input must be in the folder "folder".
def to_fasttext_dewey(folder, name, siffer):
    #deweys = ["362", "616", "346", "343", "839"]
    # ["362","616","306","346","610","343","305","839","363","320","331","307","327","341","352","658","342"]
    rootdir = folder
    label_file = open(name + '.txt', 'w')
    total_tekst = ""
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if str(file)[:5] != "meta-":
                # print(subdir)
                wiki = False
                f = open(os.path.join(subdir, file), "r+")
                tekst = f.read()
                if tekst[:9] == "__label__":
                    tekst = tekst.split(" ")
                    dewey = tekst[0]
                    tekst = " ".join(tekst[1:])
                    dewey = dewey.replace("__label__", "")
                    dewey = dewey.replace(".", "").replace("\n", "")
                    dewey = dewey[:siffer]
                    #   print (dewey)

                    wiki = True

                else:
                    f = open(os.path.join(subdir, "meta-" + str(file)), "r+")
                    for line in f.readlines():
                        if "dewey:::" in line:
                            dewey = line.split(":::")[1]

                            found = dewey.replace(".", "").replace("\n", "")

                            dewey = found[:siffer]
                            wiki = False
                            # print (dewey)

                # if found not in deweys:
                #     found="outside"
                # else:
                #     found="inside"





                # print(found)
                file_name = os.path.join(subdir, file)
                if len(dewey) != 3:
                    print(dewey)
                    print(len(dewey))
                    dewey.strip()
                if len(dewey) != 3:
                    print(dewey)
                if not wiki:
                    file_name_text = file_name.replace("meta-", "")
                    tekst_fil = open(file_name_text, "r+")
                    tekst = tekst_fil.read()
                    tekst = tekst.split(" ")
                    tekst = " ".join(tekst)
                    total_tekst += '__label__' + dewey + ' ' + tekst + '\n'
                else:
                    total_tekst += "__label__" + dewey + " " + tekst + '\n'
                    # tekst2=tekst[int(len(tekst)/2):]
                    # tekst=tekst[:int(len(tekst)/2)]


                    # total_tekst+='__label__'+found+' '+tekst2+'\n'

    label_file.write(total_tekst)


# This function takes text file(originalname) and creates a test set and training set
# with a split based on test_percentage and minimum number of articles needed per deweynumber.
def create_test_and_training_set(original_name, test_name, training_name, test_percentage, min_art):
    articles = open(original_name + '.txt', "r")
    articles = articles.readlines()
    dewey_dict = {}
    training_set = []
    test_set = []

    for article in articles:
        dewey = article.partition(' ')[0].replace("__label__", "")
        if " " in dewey:
            print(dewey)
        if dewey in dewey_dict:
            dewey_dict[dewey].append(article)
        else:
            dewey_dict[dewey] = [article]

    deweys = 0
    for key in dewey_dict.keys():
        temp = dewey_dict[key]

        # temp=temp[:1000]
        shuffle(temp)

        if len(temp) >= min_art:
            if len(temp) > 1:
                split = max(1, math.floor(len(temp) * test_percentage))
                test_set.extend(temp[:split])
                training_set.extend(temp[split:])
                # print(key)
                deweys += 1
            else:
                training_set.extend(temp)
    print("Ulike deweys:{}".format(deweys))
    shuffle(training_set)
    print(len(training_set))
    print(len(test_set))

    # shuffle(test_set)
    # training=[]
    # for i in training_set:
    #     temp_dewey=i.split(" ")[0]
    #     training.append(temp_dewey+i[int(len(i)/2):])
    #     temp=i[:int(len(i)/2)]+"\n"
    #     training.append(temp)
    #
    training = "".join(training_set)
    # training=" ".join(training)
    # test = []
    # for i in test_set:
    #     temp_dewey = i.split(" ")[0]
    #     test.append(temp_dewey + i[int(len(i) / 2):])
    #     temp = i[:int(len(i) / 2)] + "\n"
    #     test.append(temp)


    test = "".join(test_set)

    f = open(training_name + ".txt", "w")
    f.write(training)
    f = open(test_name + ".txt", "w")
    f.write(test)


def create_list_titles_fasttext(folder, pickle_name):
    rootdir = folder
    total_dewey = []
    total_tekst = []
    total_tittel = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if str(file)[:5] == "meta-":
                # print(subdir)
                f = open(os.path.join(subdir, file), "r+")
                for line in f.readlines():
                    if "tittel:::" in line and "undertittel:::" not in line:
                        tittel = line.split(":::")[1].replace("\n", "")
                        tittel = important_words(tittel)
                        tittel += "\n"
                    if "dewey:::" in line:
                        dewey = line.split(":::")[1].replace("\n", "")

                found = dewey.replace(".", "-").replace("\n", "")
                found = found[:3]

                # print(found)
                file_name = os.path.join(subdir, file)
                file_name_text = file_name.replace("meta-", "")
                tekst_fil = open(file_name_text, "r+")
                tekst = tekst_fil.read()
                tekst += "\n"

                total_dewey.append(found)
                total_tittel.append(tittel)
                total_tekst.append(tekst)

    with open(pickle_name + ".pickle", 'wb') as f:
        pickle.dump([total_dewey, total_tittel, total_tekst], f)


# for i in range(1):
#     to_fasttext_dewey("training_"+str(i),"training_"+str(i))
#     to_fasttext_dewey(""+str(i),"test_"+str(i))

# to_fasttext_dewey("hypotesetesting/test","test_stacking")
# create_list_titles_fasttext("hypotesetesting/test","stacking")
# min_art=[1,2,5,10,100]
# for i in min_art:
navn = "only_wiki_min100"
to_fasttext_dewey("/home/ubuntu/Downloads/wiki_data", navn, 3)  # 5inside_vs_outside2
#create_test_and_training_set(navn, navn + "_test", navn + "_training", 0.2, 100)

# to_fasttext_dewey("data/data_dewey/test_set","dewey_test_set2")
# to_fasttext_dewey("data/data_dewey/training_set","dewey_training_set2")

