__author__ = 'Reynaldo'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import  collections

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

#open file masuk ke kalimat
for x in range(1,601):
    f = open("doc/test/tweet%d.txt" %x , "r+" )
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
h = X_train_counts.toarray()
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'pendidikan'))

#Weighting
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


Y = []

for x in range(0,1400):
    if x < 400 :
        Y.append(0)
    elif x < 800 :
        Y.append(1)
    elif x < 1050:
        Y.append(2)
    else:
        Y.append(3)


#cluster dengan kmeans
random_state = 10000
km = KMeans(n_clusters=4, random_state=random_state)
patch_size = (20, 20)
km.fit(X_train_tfidf)
clus = []
for z in km.labels_:
    clus.append(z)
count = 0
f = open('doc/cluster.txt', 'r+')
for x in clus:
    y = str(x)
    value = y + ' '
    f.write(value)
f.close()
print(clus)
iterasi = len(clus) - 600
kelompok1 = []
kelompok2 = []
kelompok3 = []
kelompok4 = []

WordBank = [[] for x in range(0,4)]

for index in range(0,iterasi):
    if clus[index] == 0 :
        kelompok1.append(kalimat[index])
    elif clus[index] == 1 :
        kelompok2.append(kalimat[index])
    elif clus[index] == 2 :
        kelompok3.append(kalimat[index])
    else:
        kelompok4.append(kalimat[index])

print(kelompok1)
print(kelompok2)
print(kelompok3)
print(kelompok4)


#bank kata dari setiap kelompok
WordBank = [[] for x in range(0,4)]
for i in range(len(WordBank)):
    if i == 0:
        for j1 in range(len(kelompok1)):
            WordBank[i] += kelompok1[j1].split()
    elif i == 1:
        for j2 in range(len(kelompok2)):
            WordBank[i] += kelompok2[j2].split()
    elif i == 2:
        for j3 in range(len(kelompok3)):
            WordBank[i] += kelompok3[j3].split()
    else:
        for j4 in range(len(kelompok4)):
            WordBank[i] += kelompok4[j4].split()

for i in range (len(WordBank)):
    WordBank[i] = sorted(list(set(WordBank[i])))

#Extending Document Term
loop = []
for i in range(len(WordBank)):
    loop.append(WordBank[i]);


Matrix1 = [[] for x in range(len(loop[0]))]
Matrix2 = [[] for x in range(len(loop[1]))]
Matrix3 = [[] for x in range(len(loop[2]))]
Matrix4 = [[] for x in range(len(loop[3]))]

temp31 = [[] for x in range(len(loop[0]))]
temp32 = [[] for x in range(len(loop[1]))]
temp33 = [[] for x in range(len(loop[2]))]
temp34 = [[] for x in range(len(loop[3]))]




#split kalimat menjadi kata pada setiap dokumen untuk kelompok 1
for i in range(len(kelompok1)):
    word = kelompok1[i]
    Matrix1[i] = word.split()
#split kalimat menjadi kata pada setiap dokumen untuk kelompok 2
for i in range(len(kelompok2)):
    word = kelompok2[i]
    Matrix2[i] = word.split()
#split kalimat menjadi kata pada setiap dokumen untuk kelompok 3
for i in range(len(kelompok3)):
    word = kelompok3[i]
    Matrix3[i] = word.split()
#split kalimat menjadi kata pada setiap dokumen untuk kelompok 2
for i in range(len(kelompok4)):
    word = kelompok4[i]
    Matrix4[i] = word.split()

#menyimpan kata pada temp3 untuk setiap elemen WordBank
for i in range (len(WordBank[0])):
    for j in range (len(kelompok1)):
        for k in range (len(Matrix1[j]) - 1):
            if WordBank[0][i] == Matrix1[j][k] :
                temp31[i].append(Matrix1[j][k+1])


#menyimpan kata pada temp3 untuk setiap elemen WordBank
for i in range (len(WordBank[1])):
    for j in range (len(kelompok2)):
        for k in range (len(Matrix2[j]) - 1):
            if WordBank[1][i] == Matrix2[j][k] :
                temp32[i].append(Matrix2[j][k+1])


#menyimpan kata pada temp3 untuk setiap elemen WordBank
for i in range (len(WordBank[2])):
    for j in range (len(kelompok3)):
        for k in range (len(Matrix3[j]) - 1):
            if WordBank[2][i] == Matrix3[j][k] :
                temp33[i].append(Matrix3[j][k+1])


#menyimpan kata pada temp3 untuk setiap elemen WordBank
for i in range (len(WordBank[3])):
    for j in range (len(kelompok4)):
        for k in range (len(Matrix4[j]) - 1):
            if WordBank[3][i] == Matrix4[j][k] :
                temp34[i].append(Matrix4[j][k+1])



Matrix1 = [x for x in Matrix1 if x != []]
Matrix2 = [x for x in Matrix2 if x != []]
Matrix3 = [x for x in Matrix3 if x != []]
Matrix4 = [x for x in Matrix4 if x != []]

storage1 = [[] for x in range(len(temp31))] #storage untuk menyimpan hasil perhitungan pertemuan kata
storage2 = [[] for x in range(len(temp32))] #storage untuk menyimpan hasil perhitungan pertemuan kata
storage3 = [[] for x in range(len(temp33))] #storage untuk menyimpan hasil perhitungan pertemuan kata
storage4 = [[] for x in range(len(temp34))] #storage untuk menyimpan hasil perhitungan pertemuan kata

kata_tambah1 = [[] for x in range(len(temp31))]
kata_tambah2 = [[] for x in range(len(temp32))]
kata_tambah3 = [[] for x in range(len(temp33))]
kata_tambah4 = [[] for x in range(len(temp34))]


#menentukan kata paling tinggi
for x in range(len((temp31))):
    counts = collections.Counter(temp31[x])
    mostCommon = counts.most_common(1)
    storage1[x] = counts
    kata_tambah1[x] = mostCommon

for x in range(len((temp32))):
    counts = collections.Counter(temp32[x])
    mostCommon = counts.most_common(1)
    storage2[x] = counts
    kata_tambah2[x] = mostCommon

for x in range(len((temp33))):
    counts = collections.Counter(temp33[x])
    mostCommon = counts.most_common(1)
    storage3[x] = counts
    kata_tambah3[x] = mostCommon

for x in range(len((temp34))):
    counts = collections.Counter(temp34[x])
    mostCommon = counts.most_common(1)
    storage4[x] = counts
    kata_tambah4[x] = mostCommon



print(Matrix1)
print(Matrix2)
print(Matrix3)
print(Matrix4)

print('Banyaknya Bank Kata Cluster 0 : ', len(temp31))
print('Bank Kata | Kata Tambah')
for x in range(len(temp31)):
    print(WordBank[0][x] , ' | ' , kata_tambah1[x])

print('Banyaknya Bank Kata Cluster 1: ', len(temp32))
print('Bank Kata | Kata Tambah')
for x in range(len(temp32)):
    print(WordBank[1][x] , ' | ' , kata_tambah2[x])

print('Banyaknya Bank Kata Cluster 2: ', len(temp33))
print('Bank Kata | Kata Tambah')
for x in range(len(temp33)):
    print(WordBank[2][x] , ' | ' , kata_tambah3[x])

print('Banyaknya Bank Kata Cluster 3: ', len(temp34))
print('Bank Kata | Kata Tambah')
for x in range(len(temp34)):
    print(WordBank[3][x] , ' | ' , kata_tambah4[x])


train = []
for i in range(0,1400):
    train.append(kalimat[i])

print(len(train))

#VSM
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train)
h = X_train_counts.toarray()
print(X_train_counts.shape)

#Weighting
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

#classify
clf = svm.LinearSVC()
clf.fit(X_train_tfidf, Y)

test = []

for i in range(1400,2000):
    test.append(kalimat[i])

WordBankStop2 = test
nul = ''
for x in range(len(test)):
    word = test[x]
    WordBankStop2[x] = word.split()
    WordBankStop2[x] = remove_stop_words(WordBankStop2[x],StopWord)
    WordBankStop2[x] = list(filter(lambda x: x!= nul, WordBankStop2[x]))
    for y in range (len(WordBankStop2[x])):
        if len(WordBankStop2[x][y]) < 3 :
            WordBankStop2[x][y] = ''
        elif WordBankStop2[x][y] == nul:
            WordBankStop2[x][y] = ''
    test[x] = ' ' . join(WordBankStop2[x])

Y_test = []
for x in range(len(test)):
    if x < 150 :
        Y_test.append(0)
    elif x < 300 :
        Y_test.append(1)
    elif x < 450:
        Y_test.append(2)
    else:
        Y_test.append(3)

counter = 1399
j1 = 0
j2 = 0
j3 = 0
j4 = 0
Matrix21 = [[] for x in range(len(test))]
Matrix22 = [[] for x in range(len(test))]
Matrix23 = [[] for x in range(len(test))]
Matrix24 = [[] for x in range(len(test))]


print(test[300])
kata = []
for i in range(0,len(test)):
    if clus[counter] == 0:
        word = test[i]
        kata = word.split()
        for j in range (len(kata)):
            for k in range(len(temp31)):
                if kata[j] == WordBank[0][k]:
                    if kata_tambah1[k]:
                        kata.append(kata_tambah1[k][0][0])
        test[i] = ' ' . join(kata)
    elif clus[counter] == 1:
        word = test[i]
        kata = word.split()
        for j in range (len(kata)):
            for k in range(len(temp32)):
                if kata[j] == WordBank[1][k]:
                    if kata_tambah2[k]:
                        kata.append(kata_tambah2[k][0][0])
        test[i] = ' ' . join(kata)
    elif clus[counter] == 2:
        word = test[i]
        kata = word.split()
        for j in range (len(kata)):
            for k in range(len(temp33)):
                if kata[j] == WordBank[2][k]:
                    if kata_tambah3[k]:
                        kata.append(kata_tambah3[k][0][0])

        test[i] = ' ' . join(kata)
    else:
        word = test[i]
        kata = word.split()
        for j in range (len(kata)):
            for k in range(len(temp34)):
                if kata[j] == WordBank[3][k]:
                    if kata_tambah4[k] :
                        kata.append(kata_tambah4[k][0][0])
        test[i] = ' ' . join(kata)


print(kata_tambah1)
print(kata_tambah2)
print(kata_tambah3)
print(kata_tambah4)
print(counter)


#VSM datatest
X_new_counts = count_vect.transform(test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print('Hasil Prediksi :')
print(predicted)

counter_pend = 0;
counter_pol= 0
counter_wrong = 0
for x in predicted:
    if x == 0 :
        counter_pend+=1
    elif x == 1:
        counter_pol +=1
    else :
        counter_wrong+=1


label = ['Pendidikan' , 'Politik' , 'Teknologi', 'Olahraga']

print(metrics.classification_report(Y_test, predicted , target_names=label))
print('Cnfusion Matrix')
print(metrics.confusion_matrix(Y_test, predicted))
print('Akurasi Algoritma = ' , accuracy_score(Y_test, predicted))