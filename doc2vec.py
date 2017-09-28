import gensim

# https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1
# Dette scriptet fungerer ikke slik det er per nå.
def get_articles(original_name):
    articles=open(original_name+'.txt',"r")
    articles=articles.readlines()
    dewey_array = []
    doc_labels = []
    dewey_dict = {}
    for article in articles:
        dewey=article.partition(' ')[0].replace("__label__","")
        article_label_removed = article.replace("__label__"+dewey,"")
        doc_labels.append(article_label_removed)
        dewey_array.append(dewey)
        if dewey in dewey_dict:
            dewey_dict[dewey].append(article_label_removed)
        else:
            dewey_dict[dewey]=[article_label_removed]

    return dewey_array, doc_labels


class DocIterator(object):
    def __init__(self, doc_list, label_list):
        self.label_list = label_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.TaggedDocument(words=doc.split(),tags = [self.label_list[idx]])

def find_similar_docs (text, model):
    #model_test = gensim.models.Doc2Vec(size = 300, window = 10, min_count = 5, workers = 11, alpha = 0.025, min_alpha = 0.025 )
    #model_test.build_vocab

    return model.most_similar(text)

if __name__ == '__main__':
    deweys, docs = get_articles("full_text_non_stemmed")
    print(deweys)
    it = DocIterator(docs, deweys)

    model = gensim.models.Doc2Vec(size = 300, window = 10, min_count = 5, workers = 11, alpha = 0.025, min_alpha = 0.025)
    model.build_vocab(it,)

    model.alpha -=0.002
    model.min_alpha = model.alpha
    model.train(it, total_examples= model.corpus_count, epochs = 100)
    model.save("doc2vec_dir/100epoch/doc2vec_100.model")

    # model = gensim.models.Doc2Vec.load("doc2vec_dir/doc2vec.model")
    # with open("test.txt","r") as text_file:
    #     text = text_file.read().replace('\n','')
    # for word in text.split():
    #     try:
    #         print(model[word.lower()])
    #     except KeyError:
    #         print("Fant ikke ordet i vokabularet")

    #print(find_similar_docs(["genetikk"],model))


