
import pandas as pd
import numpy as np
import glob

category = ['badminton', 'cricket data', 'soccer', 'tennis']


result = []
extension = 'txt'


result = []
for i in category:
#     print(i)
    a = glob.glob('document_classification/{}/*{}'.format(i,extension))
    result.append(a)


test = glob.glob('document_classification/{}/*{}'.format('test dataset',extension))

data = []
train_label = []
for r in result:
    for fname in r:
        l = fname.split('/')
#         print(l)
        label = l[1]
        f = open(fname)
        a = f.read()
        #print(a)
        data.append([a, label])
        train_label.append(label)
        f.close()


test_data = []
test_label = []
for fname in test:
    l = fname.split('/')
    l = l[2].split(':')[0]
#     print(l)
    label = l
    f = open(fname)
    a = f.read()
#     print(a)
    test_data.append([a,label])
    test_label.append(label)
    f.close()


# In[212]:


# type(data)
test_data = np.array(test_data)

import re
def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
    return text.lower()
data_new = []
test_new = []
for d in data:
    data_new.append([text_cleaner(d[0]),d[1]])
for d in test_data:
    test_new.append([text_cleaner(d[0]),d[1]])
    
data = data_new
test_data = test_new 
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for i in symbols:
    data = np.char.replace(data, i, ' ')
    test_data = np.char.replace(test_data, i, ' ')
    
data = np.char.replace(data, "'", "")
test_data = np.char.replace(test_data, "'", "")



# from nltk.tokenize import word_tokenize
# tokens_list = []
# for file in data:
#     text = file[0]
#     tokens = word_tokenize(text)
#     tokens_list.append(tokens)




# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# stop_words.add(',')

# filtered_tokens_list = []
# for t in tokens_list:
#     filtered_tokens = []
#     for w in t:
#         if w not in stop_words:
#             filtered_tokens.append(w)
#     filtered_tokens_list.append(filtered_tokens)


from sklearn.feature_extraction.text import TfidfVectorizer

# new_tokens_list = []
# for tokens in filtered_tokens_list:
#     vectorizer_x = TfidfVectorizer(max_features=10000)
#     tokens = vectorizer_x.fit_transform(tokens).toarray()
#     new_tokens_list.append(tokens)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

train_data = data  # Documents
# test_set = ["The sun in the sky is bright."]  # Query
stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
#print vectorizer
transformer = TfidfTransformer()
#print transformer

trainVectorizerArray = vectorizer.fit_transform(train_data.ravel()).toarray()
testVectorizerArray = vectorizer.transform(test_data.ravel()).toarray()


# testVectorizerArray = vectorizer.transform(test_set).toarray()
# transformer.fit(trainVectorizerArray)
# tfidf_train = transformer.transform(trainVectorizerArray)
# tfidf_test = transformer.transform(test_data.ravel())

# print(trainVectorizerArray)
# print(testVectorizerArray)



tfidf_train = TfidfTransformer().fit_transform(trainVectorizerArray)
tfidf_test = TfidfTransformer().fit_transform(testVectorizerArray)



train_new = tfidf_train.todense()
test_new = tfidf_test.todense()



delete = []
for i in range(0,90):
    delete.append(2*i+1)
train_new = np.delete(train_new,delete,axis=0)
delete = []
for i in range(0,36):
    delete.append(2*i+1)
test_new = np.delete(test_new,delete,axis=0)


c = []
b = []
s = []
t = []
for i in range(0,90):
    l = train_label[i]
    if l == 'badminton':
        b.append(np.squeeze(np.asarray(train_new[i:i+1])))
    if l == 'cricket data':
        c.append(np.squeeze(np.asarray(train_new[i:i+1])))
    if l == 'soccer':
        s.append(np.squeeze(np.asarray(train_new[i:i+1])))
    if l == 'tennis':
        t.append(np.squeeze(np.asarray(train_new[i:i+1])))



# k = train_new[0:1]
# print(np.squeeze(np.asarray(k)))
# b = np.array(b)
# c = np.array(c)
# s = np.array(s)
# t = np.array(t)
b = np.matrix(b)
c = np.matrix(c)
s = np.matrix(s)
t = np.matrix(t)
type(b)
print(train_new)


similarity_c = []
similarity_b = []
similarity_s = []
similarity_t = []
from sklearn.metrics.pairwise import linear_kernel
l = len(b)
for i in range(0,l):
    cosine_similarities = linear_kernel(b[i], b).flatten()
    similarity_b.append(cosine_similarities)
l = len(c)
for i in range(0,l):
    cosine_similarities = linear_kernel(c[i], c).flatten()
    similarity_c.append(cosine_similarities)

l = len(s)
for i in range(0,l):
    cosine_similarities = linear_kernel(s[i], s).flatten()
    similarity_s.append(cosine_similarities)

l = len(t)
for i in range(0,l):
    cosine_similarities = linear_kernel(t[i], t).flatten()
    similarity_t.append(cosine_similarities)


import math
def getsum(M):
    s = 0
    for row in range (len(M)):
        for col in range(len(M[0])):
            s = s + M[row][col]

    return s

b_norm = math.sqrt(getsum(similarity_b)/(len(similarity_b)*len(similarity_b)))
c_norm = math.sqrt(getsum(similarity_c)/(len(similarity_c)*len(similarity_c)))
s_norm = math.sqrt(getsum(similarity_s)/(len(similarity_s)*len(similarity_s)))
t_norm = math.sqrt(getsum(similarity_t)/(len(similarity_t)*len(similarity_t)))

print(t_norm)


l = len(test_new)
label_test = []
for i in range(0,l):
    sim_b = linear_kernel(test_new[i],b).flatten()
    sim_c = linear_kernel(test_new[i],c).flatten()
    sim_s = linear_kernel(test_new[i],s).flatten()
    sim_t = linear_kernel(test_new[i],t).flatten()
    
    val_b = (sum(sim_b)/len(sim_b))/b_norm
    val_c = (sum(sim_c)/len(sim_c))/c_norm
    val_s = (sum(sim_s)/len(sim_s))/s_norm
    val_t = (sum(sim_t)/len(sim_t))/t_norm
#     print(val_b,val_c,val_s,val_t)
    val = [val_b,val_c,val_s,val_t]
    if max(val) == val_b:
        label_test.append('badminton')
    if max(val) == val_c:
        label_test.append('cricket data')
    if max(val) == val_s:
        label_test.append('soccer')
    if max(val) == val_t:
        label_test.append('tennis')
    

data = {'Original':test_label,'Predicted':label_test}
df = pd.DataFrame(data)



similarity_matrix = []
from sklearn.metrics.pairwise import linear_kernel
for i in range(0,126):
    cosine_similarities = linear_kernel(tfidf_new[i], tfidf_new).flatten()
    similarity_matrix.append(cosine_similarities)

len(similarity_matrix[0])


X = []
for i in range(0,90):
    X.append(np.squeeze(np.asarray(train_new[i:i+1])))
Y = []
for i in range(0,36):
    Y.append(np.squeeze(np.asarray(train_new[i:i+1])))
X = np.asarray(X)
Y = np.asarray(Y)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels_ = kmeans.labels_


print(len(b),len(c),len(s),len(t))
labels_


Y_pred = kmeans.predict(Y)
Y_pred

from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
Y = encoder.fit_transform(train_label)
# print(Y)
nb = MultinomialNB()
nb.fit(train_new,Y)


nb.predict(test_new)

Ytest_actual = encoder.transform(test_label)
print(Ytest_actual)

