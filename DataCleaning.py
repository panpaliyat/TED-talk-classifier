import pandas as pd
import sparse as sparse
from numpy import array
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt


ted_main = pd.read_csv('C:/Users/tusha/Desktop/AI/Datasets/TED/backup/ted-talks/ted_main.csv')
ted_transcript = pd.read_csv('C:/Users/tusha/Desktop/AI/Datasets/TED/backup/ted-talks/transcripts.csv')
ted_final = ted_transcript.merge(ted_main, on='url')

stemmer = SnowballStemmer('english')
words = stopwords.words('english')

# We only consider the columns transcript which is a feature and tags which is a target variable
col = ['transcript', 'tags']
ted_final = ted_final[col]

Y = ted_final['tags']

X = ted_final['transcript']

documents = []
filtered_tags = []

for row in Y:
    for tag in row.replace("'", "").replace('[', '').replace(']', '').split(', '):
        filtered_tags.append(tag)

unique_tags = array(filtered_tags)

tag, count = np.unique(unique_tags, return_counts=True)

tag_new = np.array([])
count_new = np.array([])

for i in range(tag.size):
    if count[i] > 300:
        tag_new = np.append(tag_new, tag[i])
        count_new = np.append(count_new, count[i])

# for i in range(tag_new.size):
#      print(tag_new[i], ":", count_new[i])

# print("*******************************************")
# print("Total tags : ", tag.size)
# print("After processing :", tag_new.size)
# print("*******************************************")

tags = []
for index, row in ted_final.iterrows():

    list1 = row['tags'].replace("'", "").replace('[', '').replace(']', '').split(', ')
    list2 = [value for value in list1 if value in tag_new]

    if not list2:
        ted_final.drop(index, inplace=True)
    else:
        ted_final.loc[index, 'tags'] = list2
        tags.append(list2)

# print(ted_final['tags'])

ted_final['cleaned'] = ted_final['transcript'].apply(lambda l: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", l).split() if i not in words]).lower())


mlb = MultiLabelBinarizer()
mlb_output = mlb.fit_transform(ted_final['tags'])

# print(type(mlb_output))
# print(mlb_output.shape)

# print(ted_final.shape)

i = 0
for category in tag_new:
    ted_final[category] = mlb_output[:, i]
    i += 1


# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
# features = tfidf.fit_transform(ted_final.transcript).toarray()
# labels = ted_final.tags
# print(features.shape)
#
# N = 4
# for tag in tag_new:
#     features_chi2 = chi2(features, labels == tag)
#     indices = np.argsort(features_chi2[0])
#     feature_names = np.array(tfidf.get_feature_names())[indices]
#     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#     # print("# '{}':".format(Product))
#     print("#########", tag, "############")
#     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# print("Original rows : ", ted_final['tags'].size)
# print("After Dropping : ", tag_final.size)




# print(ted_final.shape)

# print(mlb_output)

# arr = sparse.coo_matrix(mlb_output)
# ted_final['new_tags'] = arr.toarray().tolist()
#
train, test = train_test_split(ted_final, random_state=42, test_size=0.33, shuffle=True)

X_train = train.cleaned
X_test = test.cleaned
print(X_train.shape)
print(X_test.shape)

NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None)))])

for category in tag_new:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_test)
    print("Prediction: ", prediction)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    # print(prediction)