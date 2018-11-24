from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import nltk
import re

nltk.download('stopwords')

ted_transcript = pd.read_csv("C:/Users/tusha/Desktop/AI/Datasets/TED/transcripts.csv")

# print(ted_transcript['transcript'])

documents = []
X = ted_transcript['transcript']
y = ted_transcript['tags']

for sentence in range(0, len(X)):
    # Remove all the special characters form the sentence
    document = re.sub(r'\W', ' ', str(X[sentence]))

    # Removed single letters from the sentence
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    documents.append(document)

# print(documents)

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# print(X.shape)

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))






# Step 1 : Data set Preparation
# print(ted_main['comments'])

# Lets split the data into training and testing
# train_x, test_x, train_y, test_y = model_selection.train_test_split(ted_transcript['transcript'], ted_transcript['tags'])
# encoder = preprocessing.LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# test_y = encoder.fit_transform(test_y)
#
# # Step 2 : Feature Engineering
# count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# count_vector.fit(ted_transcript['transcript'])
#
# xtrain_count = count_vector.transform(train_x)
# xtest_count = count_vector.transform(test_x)
#
# tfidf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
#
# tfidf_vector.fit(ted_transcript['transcript'])
#
# xtrain_tfidf = tfidf_vector.transform(train_x)
# xtest_tfidf = tfidf_vector.transform(test_x)


























# nltk.download()
# data_trasncript =  pd.read_csv("C:/Users/tusha/Desktop/AI/Datasets/TED/transcripts.csv")

# print(data_main.columns)

# print(data_main.info())

# print(data_main.describe())

# print(len(data_main['event'].unique()))
# print(len(data_main['main_speaker'].unique()))
# print(data_trasncript.columns)


# vector = TfidfVectorizer()
# response = vector.fit_transform([S1, S2])
#
# print(vector.get_feature_names())
#
# print("Stop words : ",vector.get_stop_words())

# vector = TfidfVectorizer()
# vector.fit_transform(data_main.transcript)
#
# print(vector.get_stop_words())
# print(response)

# stop_words = set(stopwords.words('english'))
# words = word_tokenize(data_main.transcript[0])

# print(words)
# print("Stop Words : ",stop_words)