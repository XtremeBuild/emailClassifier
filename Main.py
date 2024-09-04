import string
import numpy as np
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

df = pd.read_csv('spam_ham_dataset.csv')
df['tect'] = df['text'].apply(lambda x: x.replace('\r\n',' '))




stemmer = PorterStemmer()
corpus = []
stopwords_set = set(stopwords.words('english'))

for i in range(0, len(df)):
    text = df['text'][i]
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if not word in stopwords_set]

    text = ' '.join(text)
    corpus.append(text)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus).toarray()
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_jobs=1)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))   