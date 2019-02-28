import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#Creating Datasets
funk = pd.read_csv('funk.csv', sep='\t', names=['lyrics'])
bossa = pd.read_csv('bossa_nova.csv', sep='\t', names=['lyrics'])
gospel = pd.read_csv('gospel.csv', sep='\t', names=['lyrics'])
sertanejo = pd.read_csv('sertanejo.csv', sep='\t', names=['lyrics'])

#Adding genre collum to datasets
genre1 = ["funk" for x in range(801)]
genre2 = ["bossa nova" for x in range(801)]
genre3 = ["gospel" for x in range(801)]
genre4 = ["sertanejo" for x in range(801)]
funk['genre'] = genre1
bossa['genre'] = genre2
gospel['genre'] = genre3
sertanejo['genre'] = genre4

#Merging and shuffling Datasets
frames = [funk, bossa, gospel, sertanejo]
dataset = pd.concat(frames)
dataset = dataset.sample(frac=1).reset_index(drop=True)


#Text Processing:
# -Removes punctuation
# -Removes stopwords (Portuguese)
# -Return clean lyrics
def text_Processing(lyrics):
    nopunc = [char for char in lyrics if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    clean_lyrics = [word for word in nopunc.split() if word.lower() not in stopwords.words('portuguese')]

    return clean_lyrics


#Defining Pipeline steps
pipeline = Pipeline([
        ('bag of words', CountVectorizer(analyzer=text_Processing)),    #Defining bag of words using Text Processing steps
        ('tfidf', TfidfTransformer()),  #Defining tf-idf of dataset lyrics bag of words
        ('Classifier', MultinomialNB()) #Defining Multinomial Naive Bayes as the classifier method
])


def setup():
    ly_train, genre_train = dataset['lyrics'], dataset['genre']
    pipeline.fit(ly_train, genre_train)


def genre_Classifier(input):
    prediction = pipeline.predict([input])

    return prediction

