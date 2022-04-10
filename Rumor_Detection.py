import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd
import re
import nltk
import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import unicodedata

Tweets= pd.read_csv("dataset3.txt", delimiter='\t', names=['text', 'type'], encoding='latin-1')

#################################################DATA CLEANING####################################################
def remove_non_ascii(self, words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(self, words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(self, words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(self, words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(self, words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(self, words):  # Stemming
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(self, words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def Remove(self, duplicate):  ##Removing duplicate elements from a list
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list







print("First 4 lines of text")
print(Tweets.head(4))
print(Tweets.describe())
print(Tweets.info)

print("-----------------------Exploratory Data Analysis-----------------------------")
print(Tweets.describe()) #For statistics like count, freq etc
print(Tweets.shape)
print(Tweets['type'])
print(Tweets['text'].head(5))

cnt_msgs = Tweets['type'].value_counts()
print(cnt_msgs)
plt.figure(figsize=(8,4))
sns.barplot(cnt_msgs.index, cnt_msgs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type of Tweets', fontsize=12)
plt.show()

#So now we know that


#Let's begin with Feature engineering.
#1. Meta feature engineering
#2. Text based

#META FEATURE ENGINEERING
Tweets["num_words"] = Tweets["text"].apply(lambda x: len(str(x).split()))
Tweets["num_unique_words"] = Tweets["text"].apply(lambda x: len(set(str(x).split())))
Tweets["num_chars"] = Tweets["text"].apply(lambda x: len(str(x)))
eng_stopwords = set(stopwords.words("english"))
Tweets["num_stopwords"] = Tweets["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
Tweets["num_punctuations"] =Tweets['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
Tweets["num_words_upper"] = Tweets["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))


## Number of title case words in the text ##
Tweets["num_words_title"] = Tweets["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

Tweets['num_words'].loc[Tweets['num_words']>60] = 60 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='type', y='num_words', data=Tweets)
plt.xlabel('Type of Tweet', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by Tweet type", fontsize=15)
plt.show()

Tweets['num_punctuations'].loc[Tweets['num_punctuations']>20] = 20 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='type', y='num_punctuations', data=Tweets)
plt.xlabel('TWEET TYPE', fontsize=12)
plt.ylabel('Number of puntuations in text', fontsize=12)
plt.title("Number of punctuations by Type", fontsize=15)
plt.show()


###############################################
stop_words = set(stopwords.words("english"))
all_words = []
rgtok = RegexpTokenizer(r'\w+')
lem = WordNetLemmatizer()

for msg in list(Tweets[Tweets["type"] == "Rumor"]["text"]):
    for word in (rgtok.tokenize(str(msg))):
        if word not in stop_words and not word.isdigit():
            all_words.append(lem.lemmatize(word.lower()))

all_words_d = nltk.FreqDist(all_words)
most_common = pd.DataFrame(data=all_words_d.most_common(45), columns=["Word", "Count"])

fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(y="Word", x="Count", data=most_common, ax=ax, orient="h")
plt.ylabel('Most Frequent words in Rumor', fontsize=12)
plt.title('Most Frequent words in Rumor')
plt.xlabel('Count', fontsize=12)
plt.show()

###########################################3
for msg in list(Tweets[Tweets["type"] == "Not a Rumor"]["text"]):
    for word in (rgtok.tokenize(str(msg))):
        if word not in stop_words and not word.isdigit():
            all_words.append(lem.lemmatize(word.lower()))

all_words_d = nltk.FreqDist(all_words)
most_common = pd.DataFrame(data=all_words_d.most_common(45), columns=["Word", "Count"])

fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(y="Word", x="Count", data=most_common, ax=ax, orient="h")
plt.ylabel('Most Frequent words in Not a Rumor', fontsize=12)
plt.title('Most Frequent words in Not a Rumor')
plt.show()



#Let's split dataset into training & testing.
textFeatures = Tweets['text'].copy()
vectorizer = CountVectorizer("english") #Selecting features on the basis of tfidf
features = vectorizer.fit_transform(textFeatures)

features_train, features_test, labels_train, labels_test = train_test_split(features, Tweets['type'], test_size=0.3, random_state=112)
svc = SVC(kernel='linear')
svc.fit(features_train, labels_train)
pre = svc.predict(features_test)
print("-------------------Accuracy using Support Vector Machine-----------------------")
print(metrics.classification_report(labels_test,pre, target_names=['Rumor', 'Not a Rumor']))
print(accuracy_score(labels_test,pre))



import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[17]:


import numpy as np


#Now let's try this using some other classifiers so as to know which one is the best
print("-------------------Accuracy using MultiNomial Naive Bayes-----------------------")
pipeline = Pipeline([
	('Featurevectorizer',  CountVectorizer(stop_words='english')),
	('classifier',  MultinomialNB(alpha=0.1)) ]) #smoothing of 0.1

scores = []
#dividing the data in folds of 10 so as to use it for testig & training purpose.
k_Fold = KFold(n_splits=10) #Broke into 10 folds
new_data_text = numpy.asarray(Tweets['text'])
new_data_class = numpy.asarray(Tweets['type'])


for train_data, test_data in k_Fold.split(Tweets['text']):
    trainMsgs = new_data_text[train_data]
    train = new_data_class[train_data]
    testText = new_data_text[test_data]
    test = new_data_class[test_data]
    pipeline.fit(trainMsgs, train) #Fitting the model
    predictions = pipeline.predict(testText) #Predicting how accurate results the classifier will give on test data
    score = pipeline.score(testText, test) #Attaining the scores
    scores.append(score)

    score = sum(scores)/len(scores)
    print("Accuracy Score: ",accuracy_score(test,predictions))
    print(confusion_matrix(test, predictions))
    print(classification_report(test, predictions))



    cnf_matrix = confusion_matrix(test, predictions)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(cnf_matrix, classes=['Rumor', 'Not a Rumor'],
                          title='Confusion matrix of NB on word count, without normalization')
    plt.show()
    break