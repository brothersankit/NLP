# -*- coding: utf-8 -*-
"""POS_Tagging_HMM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ykLvQRdAGtuDvr4q75SYeaVltx_sZmj-

# Assignment_01: POS tagging Using Hidden Markov model 

### Rajeev Dubey 2211cs34
### Swaranjali Shanker 2211cs30
### Subir Sahu 2211cs29

>Importing Libraries
"""

from urllib.request import urlopen
from sklearn.metrics import f1_score
import numpy as np
import random
import re
import json
url = "https://drive.google.com/u/0/uc?id=1R1BLcghCh4j9Kl8_CR7MxZ4Wj57RiTxn&export=download" # Read using link
response = urlopen(url)
data_json = json.loads(response.read())

"""    >Data convertion from json file to list"""

data_json[0]
pairs=[]
for j,k in data_json:
    j=re.sub(r"[,.]","",j).lower().split() # removed . , from strings and split into words
    pairs.append([])
    for i in range(len(j)):
        if len(j)==len(k) and j[i]!='--': # removed '--' cause its making data unbalanced removing it made less data removal
            pairs[-1].append([j[i],k[i]])

pairs[0],len(pairs) # Changed Data from strings to list of lists containing tags and names

"""    >given only 36 tags out of 42 available in the dataset removing garbage tags"""

with open(r"tags.txt",'r') as f:
    lines=f.readlines()
    q=set()
    for line in lines:
        q.add(line.split()[1])

len(q)

"""    Final Tags"""

' '.join(q)

"""    >collecting tags available in the dataset using sets"""

UniqueTags=set()
UniqueWords=set()
for i in pairs:
    for ind,[j,k] in enumerate(i):
        if k not in q:
            continue
        UniqueTags.add(k)
        UniqueWords.add(j)

"""    > Shuffling the data again to make sure no Outliers"""

pairs=[]
for j,k in data_json:
    j=re.sub(r"[,.]","",j).lower().split()
    pairs.append([])
    for i in range(len(j)):
        if len(j)==len(k) and k[i] in UniqueTags:
            pairs[-1].append([j[i],k[i]])
    pairs[-1]=np.array(pairs[-1])  # using numpy format for easier indexing later on

"""    > Since we are using bigram HMM model we need atleast 3 words so filtering data for those word sequences"""

pairs=[x for x in pairs if len(x)>3] #there are some pairs that are less than 3 words

"""    >Filtering out 36 out of 42 one Tag not found in dataset"""

for i in q:
    if i not in UniqueTags:
        UniqueTags.add(i)

UniqueWords,UniqueTags=list(UniqueWords),list(UniqueTags)
len(UniqueWords),len(UniqueTags)

"""    >Mapping tags to numbers and numbers to tags"""

UniqueTags_to_numbers={i:ind for ind,i in enumerate(UniqueTags)}
UniqueWords_to_numbers={i:ind for ind,i in enumerate(UniqueWords)}
numbers_to_tags={j:i for i,j in UniqueTags_to_numbers.items()}

"""    >using tuples for bigram model"""

tags_tuples=dict()
tags_numbers=dict()
k=0
for i in UniqueTags:
    for j in UniqueTags:
        tags_tuples[(i,j)]=k
        tags_numbers[(UniqueTags_to_numbers[i],UniqueTags_to_numbers[j])]=k
        k+=1

previous=None #Three pointers used to keep track of the pre  , current and next word
current=None
next=None
pi=np.ones((36),dtype='float64') # pi initial probabilities
TMatrix1=np.ones((36,36),dtype='float64') # Transision matrix for bigram
TMatrix=np.ones((1296,36),dtype='float64') # Transition matrix for trigram
EMatrix=np.ones((36,12170),dtype='float64') # Emmision matrix
random.shuffle(pairs)
train,test=pairs[:int(len(pairs)//1.25)],pairs[int(len(pairs)//1.25):] # split data into 80:20
for i in train:
    for ind,[j,k] in enumerate(i):
        if k not in UniqueTags:
            continue
        if ind==0:
            pi[UniqueTags_to_numbers[k]]+=1
            previous=k
            continue
        if ind==1:
            current=k
            TMatrix1[UniqueTags_to_numbers[previous]][UniqueTags_to_numbers[current]]+=1
            continue
        next=k
        TMatrix1[UniqueTags_to_numbers[current]][UniqueTags_to_numbers[next]]+=1
        TMatrix[tags_tuples[(previous,current)]][UniqueTags_to_numbers[next]]+=1
        previous=current
        current=next
        EMatrix[UniqueTags_to_numbers[k]][UniqueWords_to_numbers[j]]+=1
        current=k
pi=pi/pi.sum()
TMatrix1=TMatrix1/TMatrix1.sum(axis=1)[:,np.newaxis]
TMatrix=TMatrix/TMatrix.sum(axis=1)[:,np.newaxis]
EMatrix=EMatrix/EMatrix.sum(axis=1)[:,np.newaxis]

def viterbi(words):
    for j in words:
        if j not in UniqueWords_to_numbers:
            UniqueWords_to_numbers[j]=1
    matrix=np.zeros((len(words),36))
    for i in range(len(words)):
        k1=np.argmax(matrix[i-1])
        k2=np.argmax(matrix[i-2])
        if i==0:
            matrix[i]=pi*EMatrix[:,UniqueWords_to_numbers[words[i]]]
            matrix[i]=np.log(matrix[i])
            continue
        for j in range(36):
            if i==1:
                matrix[i][j]=matrix[i-1][k1]+np.log(TMatrix1[k1][j])+np.log(EMatrix[j][UniqueWords_to_numbers[words[i]]])
                # matrix[i][j]=np.log(matrix[i][j])
            else:
                matrix[i][j]=matrix[i-1][k2]+matrix[i-1][k1]+np.log(TMatrix[tags_numbers[(k2,k1)]][j])+\
                    np.log(EMatrix[j][UniqueWords_to_numbers[words[i]]])
    resultseq=[]
    for i in np.argmax(matrix,axis=1):
        resultseq.append(numbers_to_tags[i])
    return resultseq

def viterbi_firstorder(words):
    for j in words:
        if j not in UniqueWords_to_numbers:
            UniqueWords_to_numbers[j]=1
    matrix=np.zeros((len(words),36))
    for i in range(len(words)):
        k1=np.argmax(matrix[i-1])
        if i==0:
            matrix[i]=pi*EMatrix[:,UniqueWords_to_numbers[words[i]]]
            matrix[i]=np.log(matrix[i])
            continue
        for j in range(36):
            for k in range(36):
                matrix[i][j]=max(matrix[i-1][k]+np.log(TMatrix1[k][j])+np.log(EMatrix[j][UniqueWords_to_numbers[words[i]]]),
                                            matrix[i][j] )
    resultseq=[]
    for i in np.argmax(matrix,axis=1):
        resultseq.append(numbers_to_tags[i])
    return resultseq

test[1]

result=[]
for test_sample in test:
    result.append(viterbi(test_sample[:,0]))
y_pred=[]
for i in result:
    for j in i:
        y_pred.append(j)



y_true=[]
for i in test:
    for j in i[:,1]:
        y_true.append(j)

round(f1_score(y_true,y_pred,average='macro'),2),round(f1_score(y_true,y_pred,average='micro'),2)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


#Get the confusion matrix
cm = confusion_matrix(y_true, y_pred,labels=list(set(y_pred)))+1

cm=cm/cm.sum(axis=1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=list(set(y_pred)))

from matplotlib import pyplot as plt
plt.figure(figsize=(20,20))
disp.plot()

import nltk
HMMmodel=nltk.tag.HiddenMarkovModelTagger.train(train,order=2)

x_test=[]
for i in test:
    x_test.append(i[:,0].reshape(*(i[:,0].shape),1))
y_pred=[]
for i in x_test:
    i=HMMmodel.tag_sents(i)
    for j in i:
        y_pred.append(j)

len(y_pred),len(y_true)

y_pred[:10]

y_=[]
for i in y_pred:
    y_.append(i[0][1])

round(f1_score(y_true,y_,average='macro'),2),round(f1_score(y_true,y_,average='micro'),2)

result=[]
for test_sample in test:
    result.append(viterbi_firstorder(test_sample[:,0]))
y_pred=[]
for i in result:
    for j in i:
        y_pred.append(j)

round(f1_score(y_true,y_pred,average='macro'),2),round(f1_score(y_true,y_pred,average='micro'),2)

pairs[0]

' '.join(viterbi_firstorder(pairs[6][:,0]))

' '.join([x[1] for x in pairs[6]])

' '.join(UniqueTags)

"""# Converting Tags to Four [N V A J O]"""

map=dict()
for i in UniqueTags:
    if i[0]=='V':
        map[i]='V'
    elif i[0]=='J' or i[0]=='R':
        map[i]='A'
    elif i[0]=='N':
        map[i]='N'
    else:
        map[i]='O'

for i in pairs:
    for j in range(len(i)):
        i[j][1]=map[i[j][1]]

pairs[0]

UniqueTags=list(set(map.values()))
UniqueTags_to_numbers={x:ind for ind,x in enumerate(UniqueTags)}
print(UniqueTags)
tags_tuples=dict()
tags_numbers=dict()
k=0
for i in UniqueTags:
    for j in UniqueTags:
        tags_tuples[(i,j)]=k
        tags_numbers[(UniqueTags_to_numbers[i],UniqueTags_to_numbers[j])]=k
        k+=1

len(tags_numbers)

preprev=None
prev=None
next=None
pi=np.ones((4),dtype='float64') # pi initial probabilities
TMatrix1=np.ones((4,4),dtype='float64')
TMatrix=np.ones((16,4),dtype='float64') # Transition matrix
EMatrix=np.ones((4,12170),dtype='float64') # Emmision matrix
# random.shuffle(pairs)
train,test=pairs[:int(len(pairs)//1.25)],pairs[int(len(pairs)//1.25):] # split data into 80:20
for i in train:
    for ind,[j,k] in enumerate(i):
        if k not in UniqueTags:
            continue
        if ind==0:
            pi[UniqueTags_to_numbers[k]]+=1
            preprev=k
            continue
        if ind==1:
            prev=k
            TMatrix1[UniqueTags_to_numbers[preprev]][UniqueTags_to_numbers[prev]]+=1
            continue
        next=k
        TMatrix1[UniqueTags_to_numbers[prev]][UniqueTags_to_numbers[next]]+=1
        TMatrix[tags_tuples[(preprev,prev)]][UniqueTags_to_numbers[next]]+=1
        preprev=prev
        prev=next
        EMatrix[UniqueTags_to_numbers[k]][UniqueWords_to_numbers[j]]+=1
        prev=k
pi=pi/pi.sum()
TMatrix1=TMatrix1/TMatrix1.sum(axis=1)[:,np.newaxis]
TMatrix=TMatrix/TMatrix.sum(axis=1)[:,np.newaxis]
EMatrix=EMatrix/EMatrix.sum(axis=1)[:,np.newaxis]

def viterbi(words):
    for j in words:
        if j not in UniqueWords_to_numbers:
            UniqueWords_to_numbers[j]=1
    matrix=np.zeros((len(words),4))
    for i in range(len(words)):
        k1=np.argmax(matrix[i-1])
        k2=np.argmax(matrix[i-2])
        for j in range(4):
            if i==0:
                matrix[i]=pi*EMatrix[:,UniqueWords_to_numbers[words[i]]]
                matrix[i]=np.log(matrix[i])
            elif i==1:
                matrix[i][j]=matrix[i-1][k1]+np.log(TMatrix1[k1][j])+np.log(EMatrix[j][UniqueWords_to_numbers[words[i]]])
            else:
                matrix[i][j]=matrix[i-1][k2]+matrix[i-1][k1]+np.log(TMatrix[tags_numbers[(k2,k1)]][j])+\
                    np.log(EMatrix[j][UniqueWords_to_numbers[words[i]]])
    resultseq=[]
    for i in np.argmax(matrix,axis=1):
        resultseq.append(UniqueTags[i])
    return resultseq

def viterbi_firstorder(words):
    for j in words:
        if j not in UniqueWords_to_numbers:
            UniqueWords_to_numbers[j]=1
    matrix=np.zeros((len(words),4))
    for i in range(len(words)):
        k1=np.argmax(matrix[i-1])
        k2=np.argmax(matrix[i-2])
        for j in range(4):
            if i==0:
                matrix[i]=pi*EMatrix[:,UniqueWords_to_numbers[words[i]]]
                matrix[i]=np.log(matrix[i])
            else:
                matrix[i][j]=matrix[i-1][k1]+np.log(TMatrix1[k1][j])+np.log(EMatrix[j][UniqueWords_to_numbers[words[i]]])
    resultseq=[]
    for i in np.argmax(matrix,axis=1):
        resultseq.append(UniqueTags[i])
    return resultseq

result=[]
for test_sample in test:
    result.append(viterbi(test_sample[:,0]))
y_pred=[]
for i in result:
    for j in i:
        y_pred.append(j)
y_true=[]
for i in test:
    for j in i[:,1]:
        y_true.append(j)

f"Second Order Markov Chain f1 Score={f1_score(y_true,y_pred,average='macro')}"

result=[]
for test_sample in test:
    result.append(viterbi_firstorder(test_sample[:,0]))
y_pred=[]
for i in result:
    for j in i:
        y_pred.append(j)

f"First order Markov chain f1 score={f1_score(y_true,y_pred,average='macro')}"

"""# We get Better score for 4 Hidden state because of Simplicity of the Problem

>If we use 4hidden states model becomes simple 
    >now we have larger amount of data for smaller amount of data
"""

result=[]
for test_sample in train:
    result.append(viterbi(test_sample[:,0]))
y_pred=[]
for i in result:
    for j in i:
        y_pred.append(j)
y_true=[]
for i in train:
    for j in i[:,1]:
        y_true.append(j)

f"F1 score on train data ={f1_score(y_true,y_pred,average='macro')}"

result=[]
for test_sample in train:
    result.append(viterbi_firstorder(test_sample[:,0]))
y_pred=[]
for i in result:
    for j in i:
        y_pred.append(j)
f1_score(y_true,y_pred,average='macro')

from collections import Counter
counts=Counter(np.array(y_pred)==np.array(y_true))

f'Accuracy = {counts[True]/sum(counts.values())}'

"""### Ques. Comment on the following: overall performance of 36-tag vs 4-tag model; if the
overall performance of 4-tag is better than the 36-tag model, explain with intuition
with respect to transition and emission probabilistic assumption why is such the
case?

The overall performance of the 4-tag model is better than the 36-tag model because it reduces the complexity of the model. In the 4-tag model, there are fewer tags, so there are fewer possible tag sequences and the transition probabilities are simpler to calculate. Additionally, there is less ambiguity in the observation probabilities as well, since there are fewer tags to assign to each word. This makes the model more stable and less prone to overfitting.

The transition and emission probabilities in a 4-tag HMM model are calculated based on the assumption that the tag of a word depends only on the previous tag, and the word itself. This is known as the Markov property. By reducing the number of tags, the model makes the assumption that there is less variability in the way that words are used in sentences, and the data is more likely to fit this assumption.

Overall, the 4-tag model is a trade-off between accuracy and simplicity. By reducing the number of tags, the model becomes less complex and more likely to produce accurate results, but at the cost of reduced granularity.
"""

