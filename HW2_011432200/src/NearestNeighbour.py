## Move to notebook

#Imports for code


import numpy as np
import scipy as sp
#%matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend
import nltk
from nltk.stem.lancaster import LancasterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise
vectorizer = CountVectorizer()
st = LancasterStemmer()

import heapq

from nltk.corpus import stopwords

with open("TestData/train.dat", "r") as fh:
#with open("TestData/Test/training_out.dat", "r") as fh:
    linesOfTrainData = fh.readlines()
len(linesOfTrainData)

#with open("TestData/format.dat", "r") as fh:
with open("TestData/Test/format_out.dat", "r") as fh:
    linesOfFormat = fh.readlines()
len(linesOfFormat)

#with open("TestData/test.dat", "r") as fh:
with open("TestData/Test/test_out.dat", "r") as fh:
    linesOfTest = fh.readlines()
len(linesOfTest)

stops = set(stopwords.words('english'))

######

linesOfTrainDataAfterPreProcessing = []
print ("First line before cleanup: ",linesOfTrainData[0])

for line in linesOfTrainData:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            newLine = newLine.replace(' '+w+' ', ' ') # for identifting words
    linesOfTrainDataAfterPreProcessing.append(newLine)

linesOfTrainDataAfterSteming = []
for line in linesOfTrainDataAfterPreProcessing:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            newLine = newLine.replace(w, st.stem(w)) # for stemming
    linesOfTrainDataAfterSteming.append(newLine)

# get a frequency count for all words in the Test docs
wordsInTest = {}
for d in linesOfTrainDataAfterPreProcessing:
    for w in d.split():
        if w not in wordsInTest:
            wordsInTest[w] = 1
        else:
            wordsInTest[w] += 1
print("Number of unique words: %d." % len(wordsInTest))

linesOfTrainDataAfterStemingWithWordCompression = []
for line in linesOfTrainDataAfterPreProcessing:
    newLine = line
    for w in newLine.split():
        if wordsInTest[w] < 1500:
            newLine = newLine.replace(w, ' ') # for stemming
    linesOfTrainDataAfterStemingWithWordCompression.append(newLine)



print ("First line after cleanup from test Data: ", linesOfTrainDataAfterStemingWithWordCompression[0])

# Logic to do the 1% of

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
vectorizer.fit_transform(linesOfTrainDataAfterStemingWithWordCompression)
print (len(vectorizer.vocabulary_))

# print("Before cutting :", len(vectorizer.vocabulary_))
#
# for vocab in vectorizer.vocabulary_:
#     if vectorizer.vocabulary_[vocab] < 2000:
#         del vectorizer.vocabulary_[vocab]
#
# print("After cutting :", len(vectorizer.vocabulary_))

print('-----')


######

linesOfTestDataAfterPreProcessing = []
print ("First line before cleanup from test Data: ", linesOfTest[0])

for line in linesOfTest:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            newLine = newLine.replace(' '+w+' ', ' ') # for identifting words
    linesOfTestDataAfterPreProcessing.append(newLine)

print ("Before steming: "+ linesOfTestDataAfterPreProcessing[0])

linesOfTestDataAfterSteming= []
for line in linesOfTestDataAfterPreProcessing:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            print (w +" stemed to : "+st.stem(w))
            newLine = newLine.replace(w, st.stem(w)) # for identifting words
    linesOfTestDataAfterSteming.append(newLine)

print("After steming"+ linesOfTestDataAfterSteming[0])


print ("First line after cleanup from test Data: : ", linesOfTestDataAfterSteming[0])


smatrixFromTraining = vectorizer.transform(linesOfTrainDataAfterSteming)
#sp.sparse.issparse(smatrixFromTraining)


smatrixFromTesting = vectorizer.transform(linesOfTestDataAfterSteming)

print('-----')

#smatrixFromTesting = cosine_similarity(smatrixFromTesting, dense_output=False)


#result = 1 - spatial.distance.cosine(smatrixFromTraining, smatrixFromTesting)

f = open('TestData/Test/format_out.dat', 'w')
for vt in smatrixFromTesting:
    cosineSimilarityValues=[]
    for vS in smatrixFromTraining:
        dotProduct = vt.dot(np.transpose(vS))
        lengtht = np.linalg.norm(vt.data)
        lengthS = np.linalg.norm(vS.data)

        #handle exceptions

        if lengthS!=0 and lengtht!=0 :
            cosineSimilarityValue= dotProduct/(lengtht*lengthS)
        else:
            cosineSimilarityValue= 0
        cosineSimilarityValues.append(cosineSimilarityValue)

    #kneighbours = heapq.nlargest(9, cosineSimilarityValues)
    kneighbours = sp.sparse.csr_matrix.max(cosineSimilarityValues)

    neighbourReviewTypeList = []
    neighbourReviewTypeNegative = 0
    neighbourReviewTypePositive = 0

    for neighbour in kneighbours:
            index = cosineSimilarityValues.index(neighbour.data[0])

            if linesOfTrainData[index].strip()[0] == '-':
                neighbourReviewTypeList.append("-1")
                neighbourReviewTypeNegative+=1
            elif linesOfTrainData[index].strip()[0] == '+':
                neighbourReviewTypeList.append("+1")
                neighbourReviewTypePositive+=1


    if neighbourReviewTypeNegative > neighbourReviewTypePositive:
        f.write('-1\n')
    else:
        f.write('+1\n')


#    print(len(cosineSimilarityValues));


print('---END---')




# vectorizer.fit_transform(linesOfTrainDataAfterPreProcessing)
# print (vectorizer.vocabulary_)
#
# #Sparce vectore for training
# smatrixFromTraining = vectorizer.transform(linesOfTrainDataAfterPreProcessing)
# print(smatrixFromTraining)
#
# #sparce vectore for Test
# smatrixFromTest = vectorizer.transform(linesOfTestDataAfterPreProcessing)
# print(smatrixFromTest)


# dotProduct = sp.dot(smatrixFromTraining, smatrixFromTest)
# print(dotProduct)

#cosineSimilarity = pairwise.cosine_similarity(smatrixFromTraining, smatrixFromTest, dense_output=False)
#print(cosineSimilarity)



