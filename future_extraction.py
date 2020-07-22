

__author__ = 'Reynaldo'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re


#var
kalimat = []

vectorizer = CountVectorizer(min_df=1)
print (vectorizer)

for x in range(1,87):
    f = open("doc/tweet%d.txt" %x , "r+" )
    tweet = f.readline();
    tweet = re.sub("[^a-zA-Z]+", " ", tweet)
    kalimat.append(tweet.lower())
    f.close()




X = vectorizer.fit_transform(kalimat)

print(X)

analyze = vectorizer.build_analyzer()

print(analyze("dengan teknologi acer bluelight shield di dlm acer aspire es mata kamu tidak akan mudah lelah") == (['dengan', 'teknologi', 'acer', 'bluelight', 'to', 'analyze']))



print(X.toarray())
print('\n\n\n\n\n')
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
X_2 = bigram_vectorizer.fit_transform(kalimat).toarray()
print(X_2)
print('\n\n\n\n\n')
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X_2)
print( tfidf.toarray() )