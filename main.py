import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
import string

# Create vector for all sentences from data
def createVectors(each):
    v = [0] * count
    for word in each:
        if word in dictofwords:
            v[dictofwords[word]] += 1
    vectors.append(v)

def logic(x):
    return not (x % 50 == 0)


def processTweet(word):
     
    if len(word) < 2 or word.find("http:") != -1 or word.find("www.") != -1 or re.search("[0-9]", word):
        return

    
    if re.search(":|;", word):
        if word in dictofwordscount:
            dictofwordscount[word] += 1
        else:
            dictofwordscount[word] = 0
            setofwords.add(word)
        return word

    # Apply Spell Checker
    word = spell.correction(word)
 

    # Apply Stemmer
    word = stemmer.stem(word)

    # Skip if Stop Word
    if word in stopwords.words('english'):
        return

    else:
        if word in dictofwordscount:
            dictofwordscount[word] += 1

     
        else:
            dictofwordscount.update({word: 0})
            setofwords.add(word)
        return word

def pieChart(y_test,predicted):
    uniqueOG, countsOG = np.unique(y_test.to_numpy(), return_counts=True)
    unique, counts = np.unique(predicted, return_counts=True)
    dict(zip(unique, counts))
    dict(zip(uniqueOG, countsOG))
    # Data to plot
    labels = [];sizes = []
    labelsOG = [];sizesOG = []

    for x, y in dict(zip(unique, counts)).items():
        labels.append("Positive Tweets" if x == 4 else "Negative Tweets")
        sizes.append(y)
    for x, y in dict(zip(uniqueOG, countsOG)).items():
        labelsOG.append("Positive Tweets" if x == 4 else "Negative Tweets")
        sizesOG.append(y)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10)) #ax1,ax2 refer to your two pies
    ax1.pie(sizesOG,labels = labelsOG,autopct = '%1.1f%%') #plot first pie
    ax1.title.set_text('Original :')
    ax2.pie(sizes, labels=labels,autopct='%1.1f%%')
    ax2.title.set_text('Predicted :')
    plt.show()
    return

# Read data
df = pd.read_csv("data.csv", delimiter=",", encoding='latin-1', skiprows=lambda x: logic(x))
print("Len", len(df))

table = str.maketrans('', '', string.punctuation)
tweet = df["Tweet"]
stripped = [w.translate(table) for w in tweet]
df["Tweet"] = stripped



# Trimming data
df.drop(["Date", "ID", "Query", "User"], 1, inplace=True)

# Declare Tokenizer
toknizr = TweetTokenizer(strip_handles=True, preserve_case=False, reduce_len=True)

# CONSIDER LEMMATIZATION INSTEAD OF STEMMING
# Declare Stemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

# Declare SpellChecker
spell = SpellChecker(distance=1)


# Declare Set
setofwords = set()

# Declare dict for storing counts
dictofwordscount = dict()


df["Tweet"] = df["Tweet"].apply(toknizr.tokenize)
df["origTweet"] = df["Tweet"]

df["Tweet"].apply(lambda x: list(filter(None, (list(map(processTweet, x))))))


df.drop(["Tweet"], 1, inplace=True)

# Declare a cut off freq for words
cutoffcount = 25

# Create Dictionary
count = 0
dictofwords = dict()

#dictofwords --> contains all those unique words whose frequency is greatet than the cutoff frequency and autoincrements count for each unique word
for each in setofwords:
    if dictofwordscount[each] > cutoffcount:
        dictofwords[each] = count
        count += 1

#count represents the total number of unique words in setofthewords having frequency greater than the cutoff frequency  
del setofwords

pickle.dump(dictofwords, open("dictofwords.pickle", "wb"))
pickle.dump(count, open("vectorsize.pickle", "wb"))

# Declare Vector
vectors = list()


df["origTweet"].apply(lambda x: list(map(spell.correction, x)))
df["origTweet"].apply(lambda x: list(map(stemmer.stem, x)))
df["origTweet"].apply(createVectors)
# print(df.head(30))

del dictofwords

df.drop(["origTweet"], 1, inplace=True)

features = vectors
del vectors

label = df["Polarity"]


classifier = []
accuracy = []

#splits the feature set into training  and testing data set
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.01)


#Guassian Naive bayes
print('Guassain Naive Bayes Classsifier')
model = GaussianNB()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
pickle.dump(model, open("model.pickle", "wb"))
acc= metrics.accuracy_score(y_test, predicted)
print("Accuracy:", acc )
print("Count:", count)
print("")
pieChart(y_test,predicted)
classifier.append("GaussianNB")
accuracy.append(acc)


#mutinomial naive bayes classsifier
print('Multinomial Naive Bayes Classifier ')
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
acc= metrics.accuracy_score(y_test, predicted)
print("Accuracy:", acc)
print("Count:", count)
print("")
pieChart(y_test,predicted)
classifier.append("Multinomial")
accuracy.append(acc)


#linear support vector 
print('linearSVM :')
from sklearn.svm import LinearSVC
SVM = LinearSVC()
SVM.fit(X_train, y_train)
predicted = SVM.predict(X_test)
acc= metrics.accuracy_score(y_test, predicted)

print("Accuracy:", acc)
print("Count:", count)
print("")
pieChart(y_test,predicted)
classifier.append("SVM")
accuracy.append(acc)


#KNN
print('K Nearest Neighbours : ')
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
acc= metrics.accuracy_score(y_test, predicted)

print("Accuracy:", acc)
print("Count:", count)
pieChart(y_test,predicted)
classifier.append("KNN")
accuracy.append(acc)

# plotting variation of different classifier
plt.title('Variation in Classifier')
plt.xlabel('Classifier Used')
plt.ylabel('Accuracy')

xp = classifier
yp = accuracy
plt.plot(xp, yp)
plt.show()