# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt
# %matplotlib inline

from subprocess import check_output

data = pd.read_csv('/Users/yihe/PycharmProjects/tweet_sentiment/dataset/Sentiment.csv')
data = data[['tweet','senti']]
train, test = train_test_split(data,test_size = 0.1)

train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neu = train[ train['sentiment'] == 'Neutral']
train_neu = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

# Nettroyer url, hashtag
