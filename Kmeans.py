
__author__ = 'Reynaldo'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

import re
# variabel declare
kalimat = []
StopWord =[]
WordBank =[]
W = ""
WordBankStop = []
WordBankStopTS = []
temp1 = []
ftd = []
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
for x in range(1,201):
    f = open("doc/tweet%d.txt" %x , "r+" )
    tweet = f.readline();
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    kalimat.append(tweet.lower())
    f.close()

for x in kalimat:
    print(x)

#WordBankStop
WordBankStop = kalimat

nul = ''

for x in range(len(kalimat)):
    word = kalimat[x]
    WordBankStop[x] = word.split()
    WordBankStop[x] = remove_stop_words(WordBankStop[x],StopWord)
    WordBankStop[x] = list(filter(lambda x: x!= nul, WordBankStop[x]))
    WordBankStop = [x for x in WordBankStop if x != []]
    kalimat[x] = ' ' . join(WordBankStop[x])



#WordBank
for x in WordBankStop:
    WordBank += x


#end word bank

WordBank.sort()


for i in range(len(WordBankStop)):
    WordBankStop[i] = sorted(WordBankStop[i])



print(WordBank)
print(WordBankStop)



print(WordBankStop[0])

X = ' ' . join(WordBankStop[0])


print(X)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(kalimat)
h = X_train_counts.toarray()
for y in h :
    print(y)

print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'pendidikan'))

#Weighting
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

km = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=1)
km.fit(X_train_tfidf)

print(km.labels_)