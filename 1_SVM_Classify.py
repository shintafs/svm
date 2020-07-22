from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

# variabel declare
kalimat = []
StopWord = []
WordBank = []
W = ""
WordBankStop = []
WordBankStopTS = []
temp1 = []
test = []
# inisialisasi SropWord
stopw = open("doc/stopword/StopWord.txt", "r+")

# fungsi stopword


def remove_stop_words(wordlist, stopwords):
    # ask for sentence if wordlist is empty
    if not wordlist:
        sentence = raw_input("type a sentence: ")
        wordlist = sentence.split()
    marked = []
    for t in wordlist:
        if t.lower() in stopwords:
            marked.append('')
        else:
            marked.append(t)
    return marked


SW = stopw.readlines()

for x in SW:
    StopWord.append(re.sub("[^a-zA-Z]+", "", x))
stopw.close()

# Inisialisasi WordBank dan WordBankStop

# open file masuk ke kalimat
for x in range(1, 1401):
    f = open("doc/train/tweet%d.txt" % x, "r+")
    tweet = f.readline()
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    kalimat.append(tweet.lower())
    f.close()


# WordBankStop
WordBankStop = kalimat
trashhold = kalimat
print(len(kalimat), len(WordBankStop))

nul = ''
# stopword
for x in range(len(kalimat)):
    word = kalimat[x]
    WordBankStop[x] = word.split()
    WordBankStop[x] = remove_stop_words(WordBankStop[x], StopWord)
    WordBankStop[x] = list(filter(lambda x: x != nul, WordBankStop[x]))
    for y in range(len(WordBankStop[x])):
        if len(WordBankStop[x][y]) < 3:
            WordBankStop[x][y] = ''
    kalimat[x] = ' ' . join(WordBankStop[x])


print(len(kalimat))
# VSM
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(kalimat)
h = X_train_counts.toarray()
# print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'pendidikan'))

# Weighting
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


Y = []

# label setiap twwet
for x in range(len(kalimat)):
    if x < 350:
        Y.append(0)
    elif x < 700:
        Y.append(1)
    elif x < 1050:
        Y.append(2)
    elif x < 1400:
        Y.append(3)

# classify
clf = svm.LinearSVC()
clf.fit(X_train_tfidf, Y)

counter_pend = 0
counter_pol = 0
counter_wrong = 0
for x in Y:
    if x == 0:
        counter_pend += 1
    elif x == 1:
        counter_pol += 1
    else:
        counter_wrong += 1

print(counter_pend, counter_pol, counter_wrong)
# load datatest
for x in range(1, 601):
    f = open("doc/train/tweet%d.txt" % x, "r+")
    tweet = f.readline()
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    test.append(tweet.lower())
    f.close()

WordBankStop2 = test
nul = ''
for x in range(len(test)):
    word = test[x]
    WordBankStop2[x] = word.split()
    WordBankStop2[x] = remove_stop_words(WordBankStop2[x], StopWord)
    WordBankStop2[x] = list(filter(lambda x: x != nul, WordBankStop2[x]))
    for y in range(len(WordBankStop2[x])):
        if len(WordBankStop2[x][y]) < 3:
            WordBankStop2[x][y] = ''
        elif WordBankStop2[x][y] == nul:
            WordBankStop2[x][y] = ''
    test[x] = ' ' . join(WordBankStop2[x])

Y_test = []
for x in range(len(test)):
    if x < 150:
        Y_test.append(0)
    elif x < 300:
        Y_test.append(1)
    elif x < 450:
        Y_test.append(2)
    else:
        Y_test.append(3)


# VSM datatest
X_new_counts = count_vect.transform(test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print(predicted)


print(counter_pend, counter_pol, counter_wrong)
print(Y_test)
label = ['Pendidikan', 'Politik', 'Teknologi', 'Olahraga']


print(metrics.classification_report(Y_test, predicted, target_names=label))
print("Confussion Matrix :")
print(metrics.confusion_matrix(Y_test, predicted))
print('Akurasi Algoritma = ', accuracy_score(Y_test, predicted))
print("Hasil Prediksi :")
print(predicted)

