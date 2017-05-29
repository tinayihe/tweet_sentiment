# import libraries
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt
# %matplotlib inline

from subprocess import check_output

data = pd.read_csv('/Users/yihe/PycharmProjects/tweet_sentiment/dataset/Sentiment.csv')
data = data[['text','sentiment']]
train, test = train_test_split(data,test_size = 0.1)

#classify tweets' texts
train_pos = train[train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neu = train[train['sentiment'] == 'Neutral']
train_neu = train_neu['text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg = train_neg['text']
# train_iro = train[train['sentiment'] == 'Ironique']
# train_iro = train_neg['text']

# Visualisation de feutures
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# Negation
# input: list of words
def groupe_negation(list_words):
    negative_ends = ['pas', 'jamais', 'rien']
    start_index = -1
    end_index = -1
    for idx, val in enumerate(list_words):
        if val.startwith("n'") or val == 'ne':
            start_index = idx
        if val in negative_ends:
            end_index = idx
        if start_index != -1 and end_index != -1 and start_index < end_index:
            list_words[start_index:end_index] = [' '.join(list_words[start_index:end_index])]
    negative_before_adjective = ['non', 'pas']
    for idx, val in enumerate(list_words):
        if val in negative_before_adjective and idx+1<len(list_words):
            list_words[idx:idx+1] = [' '.join(list_words[idx:idx+1])]
    return list_words

# Delete punctuation
def delete_punctuation(words):
    for idx, word in enumerate(words):
        if word.endswith(',') or word.endswith('.'):
            words[idx] = word[:-1]
    return words

# Extract feature emojicon for each catelog
def extract_emoji(words):

    return words

tweets = []
# stop words
# stopwords_set = set(stopwords.words("french"))
stopwords_set = set(stopwords.words("english"))
for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    # Nettroyer url, mention, hashtag
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    # extract emojicon


    # Groupe négation
    words_cleaned = groupe_negation(words_cleaned)

    # Delete punctuation
    words_cleaned = delete_punctuation(words_cleaned)

    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords,row.sentiment))


# Extracting word features
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

# All features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features

# training_set = nltk.classify.apply_features(extract_features,tweets)
# print(training_set)

