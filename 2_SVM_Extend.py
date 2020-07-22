__author__ = 'Reynaldo'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
import  collections
import re
from sklearn import metrics
from sklearn.metrics import accuracy_score


#var
kalimat = []
WordBankStop = []
StopWord =[]
test = []
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

#stopword inisialisasi
stopw = open("doc/stopword/StopWord.txt" , "r+" )
SW = stopw.readlines()
for x in SW:
    StopWord.append( re.sub("[^a-zA-Z]+", "", x))
stopw.close()

for x in range(1,1401):
    f = open("doc/train/tweet%d.txt" %x , "r+" )
    tweet = f.readline();
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    tweet=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','', tweet)
    kalimat.append(tweet.lower())
    f.close()

nul = ''

print(kalimat[0])
WordBankStop = kalimat
for x in range(len(kalimat)):
    word = kalimat[x]
    WordBankStop[x] = word.split()
    WordBankStop[x] = remove_stop_words(WordBankStop[x],StopWord)
    WordBankStop[x] = list(filter(lambda x: x!= nul, WordBankStop[x]))
    kalimat[x] = ' ' . join(WordBankStop[x])


WordBank = []
for x in range(len(kalimat)):
    WordBank += kalimat[x].split()


WordBank = sorted(list(set(WordBank)))


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

for x in range(len(kalimat)):
    if x < 350 :
        Y.append(0)
    elif x < 700 :
        Y.append(1)
    elif x < 1050:
        Y.append(2)
    else:
        Y.append(3)

#classify
clf =  svm.LinearSVC()
clf.fit(X_train_tfidf, Y)
print(Y)
counter_pend = 0;
counter_pol= 0
counter_wrong = 0
for x in Y:
    if x == 0 :
        counter_pend+=1
    elif x == 1:
        counter_pol +=1
    else :
        counter_wrong+=1


print(counter_pend, counter_pol, counter_wrong)
#load datatest
for x in range(1,601):
    f = open("doc/test/tweet%d.txt" %x , "r+" )
    tweet = f.readline();
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    test.append(tweet.lower())
    f.close()

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

#Extending Document Term

loop1 = len(WordBank)
loop2 = len(kalimat)
print(loop1, loop2)
loop3 = len(test)
Matrix = [[] for x in range(loop1)]
Matrix2 = [[] for x in range(loop3)]
temp3 = [[] for x in range(loop1)]

#split kalimat menjadi kata pada setiap dokumen
for i in range(len(kalimat)):
    word = kalimat[i]
    Matrix[i] = word.split()

for i in range(len(test)):
    word = test[i]
    Matrix2[i] = word.split()

#menyimpan kata pada temp3 untuk setiap elemen WordBank
for i in range (0,loop1):
    for j in range (0,loop2):
        for k in range (len(Matrix[j]) - 1):
            if WordBank[i] == Matrix[j][k] :
                temp3[i].append(Matrix[j][k+1])




storage = [[] for x in range(len(temp3))] #storage untuk menyimpan hasil perhitungan pertemuan kata
kata_tambah = [[] for x in range(len(temp3))] #menyimpan hasil kata yang paling sering bertemu, index sesuai WordBank
for x in range(len((temp3))):
    counts = collections.Counter(temp3[x])
    mostCommon = counts.most_common(1)
    storage[x] = counts
    kata_tambah[x] = mostCommon


for i in range (len(Matrix2)):
    for j in range(len(Matrix2[i])):
        for k in range (len(temp3)):
            if Matrix2[i][j] == WordBank[k]:
                if kata_tambah[k] :
                    Matrix2[i].append(kata_tambah[k][0][0])


#menghapus kata yang kurang dari 3 karakter
for i in range (len(Matrix2)):
    for j in range(len(Matrix2[i])):
        if len(Matrix2[i][j]) < 3 :
            Matrix2[i][j] = ' '

print('Banyaknya Bank Kata : ', len(temp3))
print('Bank Kata | Kata Tambah')
for x in range(len(WordBank)):
    print(WordBank[x] , ' | ' , kata_tambah[x])


for i in range (len(Matrix2)):
    if Matrix2[i]:
        test[i] = ' ' . join(Matrix2[i])

#VSM datatest
X_new_counts = count_vect.transform(test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print("Hasil prediksi :")
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
print(counter_pend, counter_pol,counter_wrong)

label = ['Pendidikan' , 'Politik' , 'Teknologi', 'Olahraga']

print(metrics.classification_report(Y_test, predicted , target_names=label))
print("COnfussion Matrix :")
print(metrics.confusion_matrix(Y_test, predicted))
print('Akurasi Algoritma = ' , accuracy_score(Y_test, predicted))