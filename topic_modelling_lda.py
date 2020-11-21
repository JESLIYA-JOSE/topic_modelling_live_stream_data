import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import pickle

nltk.download('punkt')
reviews_datasets = pd.read_csv(r'news_summary_more.csv')
def tokenize(text):
    tokens=[word for word in nltk.word_tokenize(text) if len(word) >3 ]
    return tokens

def topic_modelling(flag):#function for tokenization, training etc
    pd.set_option('display.max_colwidth', -1)
    X_train,x_test = train_test_split(reviews_datasets, test_size=0.9, random_state=111)
    # printx_test,typex_test))

    vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.75, min_df=50, max_features=10000, use_idf=False, norm=None)
    tf_vectors = vectorizer_tf.fit_transform(X_train.text)
    if(flag==1):
        lda = decomposition.LatentDirichletAllocation(n_components=10, max_iter=3, learning_method='online', learning_offset=50, n_jobs=-1, random_state=111)
        with open("lda_model.pk","wb") as f:
            pickle.dump(lda, f)
    else:
        with open("lda_model.pk","rb") as f:
            lda = pickle.load(f)
    W1 = lda.fit_transform(tf_vectors)
    H1 = lda.components_
    num_words=15
    vocab = np.array(vectorizer_tf.get_feature_names())
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
    topic_words = ([top_words(t) for t in H1])
    topics = [' '.join(t) for t in topic_words]
    colnames = ["Topic" + str(i) for i in range(lda.n_components)]
    docnames = ["Doc" + str(i) for i in range(len(X_train.text))]
    df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
    df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
    topic_important = np.argmax(df_doc_topic.values, axis=1)
    df_doc_topic['most_matched_topic'] = topic_important
    
    
    print("Log Likelihood: ", lda.score(tf_vectors))
    print("Perplexity: ", lda.perplexity(tf_vectors))
    return lda,vectorizer_tf,topics


def test_topic_model(dataframe_passed,lda,vectorizer_tf,topics):#function for testing with new data
    df=dataframe_passed
    WHold = lda.transform(vectorizer_tf.transform(df.text))
    with open("lda_model.pk","rb") as f:
        lda = pickle.load(f)
    colnames = ["Topic" + str(i) for i in range(lda.n_components)]
    docnames = ["Doc" + str(i) for i in range(len(df.text))]
    df_doc_topic = pd.DataFrame(np.round(WHold, 2), columns=colnames, index=docnames)
    topic_important = np.argmax(df_doc_topic.values, axis=1)
    df_doc_topic['most_matched_topic'] = topic_important
    return topics[int(topic_important[0])]
def main():
    data = [[0,"UK teen pleads guilty to hacking ex-Obama aides' computers    A British teenager has pleaded guilty to hacking into computers belonging to ex-US President Barack Obama's National Security Adviser Avril Haines and his Senior Science and Technology Adviser John Holdren. The teenager, who also hacked the computer of an ex-CIA Director, has admitted to 10 charges of hacking. However, the teenager's legal representative has claimed that his client is autistic.nnn"]] 
    df = pd.DataFrame (data, columns = ['no','text'])
    # print(df)
    lda,vectorizer_tf,topics=topic_modelling(0)
    print("what is this................",test_topic_model(df,lda,vectorizer_tf,topics))