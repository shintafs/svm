__author__ = 'Reynaldo'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
import re
import  collections


#var
kalimat = []
WordBankStop = []
StopWord =[]
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

for x in range(1,1201):
    f = open("doc/tweet%d.txt" %x , "r+" )
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



#Extending Document Term

loop1 = len(WordBank)
loop2 = len(kalimat)
print(loop1, loop2)
Matrix = [[] for x in range(loop1)]
temp3 = [[] for x in range(loop1)]

#split kalimat menjadi kata pada setiap dokumen
for i in range(len(kalimat)):
    word = kalimat[i]
    Matrix[i] = word.split()

#menyimpan kata pada temp3 untuk setiap elemen WordBank
for i in range (0,loop1):
    for j in range (0,loop2):
        for k in range (len(Matrix[j]) - 1):
            if WordBank[i] == Matrix[j][k] :
                temp3[i].append(Matrix[j][k+1])




Matrix = [x for x in Matrix if x != []] #menghilangkain karakter kosong pada bank kata
storage = [[] for x in range(len(temp3))] #storage untuk menyimpan hasil perhitungan pertemuan kata



kata_tambah = [[] for x in range(len(temp3))] #menyimpan hasil kata yang paling sering bertemu, index sesuai WordBank


for x in range(len((temp3))):
    counts = collections.Counter(temp3[x])
    mostCommon = counts.most_common(1)
    storage[x] = counts
    kata_tambah[x] = mostCommon





print(Matrix)
print(temp3)
print(storage)
print(kata_tambah)
print(len(WordBank), len(temp3), len(kata_tambah))


for i in range (len(Matrix)):
    for j in range(len(Matrix[i])):
        for k in range (len(temp3)):
            if Matrix[i][j] == WordBank[k]:
                if kata_tambah[k] :
                    Matrix[i].append(kata_tambah[k][0][0])


#menghapus kata yang kurang dari 3 karakter
for i in range (len(Matrix)):
    for j in range(len(Matrix[i])):
        if len(Matrix[i][j]) < 3 :
            Matrix[i][j] = ' '
Matrix = [x for x in Matrix if x != []]


print(kalimat[0], len(kalimat))
print(Matrix[0], len(Matrix))

for i in range (len(kalimat)):
    kalimat[i] = ' ' . join(Matrix[i])

print(kalimat)
#VSM
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

Y = [0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,
     1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,
     3,2,1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,1,0,0,1,2,3,3,2,1,0,2,1,1,1,3,2,1,2,3,2,1,0,3,2,1,0,0,1,2,3,3,2,3,2,1,0,
     0,1,2,3,3,2,3,2,1,0,0,1,2,3,3,2,3,2,1,0,0,1,2,3,3,2]

#classify
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_tfidf, Y)



docs_new = ['pendidikan adalah investasi masa depan', 'aku belajar untuk menyelesaikan pendidikan']
Test = docs_new
for x in range(len(docs_new)):
    word = docs_new[x]
    Test[x] = word.split()
    Test[x] = remove_stop_words(Test[x],StopWord)
    Test[x] = list(filter(lambda x: x!= nul, Test[x]))
    docs_new[x] = ' ' . join(Test[x])





X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


predicted = clf.predict(X_new_tfidf)


print(predicted)






