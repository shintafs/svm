__author__ = 'Reynaldo'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
    f = open("doc/datatest/tweet%d.txt" %x , "r+" )
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


yu = len(StopWord) -1
print(StopWord[yu])


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

vectorizer = CountVectorizer(min_df=1)

print (vectorizer)

