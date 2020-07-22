__author__ = 'Reynaldo'
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
StopWord =[]
WordBank =[]
W = ""
WordBankStop = []
WordBankStopTS = []
temp1 = []
test = []
#inisialisasi SropWord
stopw = open("doc/stopword/StopWord.txt" , "r+" )

#fungsi stopword
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
    StopWord.append( re.sub("[^a-zA-Z]+", "", x))
stopw.close()

#Inisialisasi WordBank dan WordBankStop

#open file masuk ke kalimat
for x in range(1,1401):
    f = open("doc/train/tweet%d.txt" %x , "r+" )
    tweet = f.readline();
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    kalimat.append(tweet.lower())
    f.close()


#WordBankStop
WordBankStop = kalimat
trashhold = kalimat
print(len(kalimat), len(WordBankStop))

nul = ''

for x in range(len(kalimat)):
    word = kalimat[x]
    WordBankStop[x] = word.split()
    WordBankStop[x] = remove_stop_words(WordBankStop[x],StopWord)
    WordBankStop[x] = list(filter(lambda x: x!= nul, WordBankStop[x]))
    for y in range (len(WordBankStop[x])):
        if len(WordBankStop[x][y]) < 3 :
            WordBankStop[x][y] = ''
    kalimat[x] = ' ' . join(WordBankStop[x])



print(len(kalimat))
#VSM
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(kalimat)
print(X_train_counts.shape)


y = X_train_counts.toarray()

for i in range(len(y)):
    print(y[i])
#Weighting
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

y = X_train_tfidf.toarray()

for i in range(len(y)):
    print(y[i])
