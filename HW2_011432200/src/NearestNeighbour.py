#Imports for code
import numpy as np
import scipy as sp
#%matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend
import nltk
from nltk.stem.lancaster import LancasterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise
vectorizer = CountVectorizer()
st = LancasterStemmer()

import heapq

from nltk.corpus import stopwords

#with open("TestData/train.dat", "r") as fh:
with open("TestData/Test/training_out.dat", "r") as fh:
    linesOfTrainData = fh.readlines()
len(linesOfTrainData)

# transform docs into lists of words
# docsTrain = [l.split() for l in linesOfTrainData]
# print("Train data:")
# print("Number of docs: %d." % len(docsTrain))
# print("Number of words: %d." % np.sum([len(d) for d in docsTrain]))

# Format data : open docs file and read its lines
#with open("TestData/format.dat", "r") as fh:
with open("TestData/Test/format_out.dat", "r") as fh:
    linesOfFormat = fh.readlines()
len(linesOfFormat)

# transform docs into lists of words
# docsFormat = [l.split() for l in linesOfFormat]
# print("Format data:")
# print("Number of docs: %d." % len(docsFormat))
# print("Number of words: %d." % np.sum([len(d) for d in docsFormat]))

# Test Data : open docs file and read its lines
#with open("TestData/test.dat", "r") as fh:
with open("TestData/Test/test_out.dat", "r") as fh:
    linesOfTest = fh.readlines()
len(linesOfTest)

# transform docs into lists of words
# docsTest = [l.split() for l in linesOfTest]
# print("Test data:")
# print("Number of docs: %d." % len(docsTest))
# print("Number of words: %d." % np.sum([len(d) for d in docsTest]))

# get a frequency count for all words in the Training docs
# wordsInTraining = {}
# for d in docsTrain:
#     for w in d:
#         if w not in wordsInTraining:
#             wordsInTraining[w] = 1
#         else:
#             wordsInTraining[w] += 1
# print("Number of unique words in Training: %d." % len(wordsInTraining))

# #print ("Number of is in Test: %d." % wordsInTest['is'])
# print ("Number of is in Training: %d." % wordsInTraining['is'])


# positiveReview = 0
# negativeReview = 0
# for d in docsFormat:
#     for w in d:
#         #print(w)
#         if w.strip()[0] == '-':
#             negativeReview += 1
#         elif w.strip()[0] == '+':
#             positiveReview += 1
#         break
#
# print("Positive reviews : %d." % positiveReview)
# print("Negative reviews : %d." % negativeReview)

# positiveReview = 0
# negativeReview = 0
# for d in docsTrain:
#     for w in d:
#         #print(w)
#         if w.strip()[0] == '-':
#             negativeReview += 1
#         elif w.strip()[0] == '+':
#             positiveReview += 1
#         break
#
# print("Positive reviews : %d." % positiveReview)
# print("Negative reviews : %d." % negativeReview)

#test for stopwords
# stops = set(stopwords.words('english'))
#
# wordCount = 0
# for line in linesOfTrainData:
#     for w in line.split():
#         if w.lower() not in stops:
#             wordCount+=1
#
# print(wordCount)

#Total world count
# wordCount = 0
# for line in linesOfTrainData:
#     for w in line.split():
#         wordCount+=1
#
# print(wordCount)

#test for stopwords
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


print ("First line after cleanup from test Data: ", linesOfTrainDataAfterSteming[0])


vectorizer.fit_transform(linesOfTrainDataAfterSteming)
print (vectorizer.vocabulary_)

#Sparce vectore for training
smatrixFromTraining = vectorizer.transform(linesOfTrainDataAfterSteming)
print(smatrixFromTraining)

print('-----')

smatrixFromTraining = cosine_similarity(smatrixFromTraining, dense_output=False)

######

linesOfTestDataAfterPreProcessing = []
print ("First line before cleanup from test Data: ", linesOfTest[0])

for line in linesOfTest:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            newLine = newLine.replace(' '+w+' ', ' ') # for identifting words
    linesOfTestDataAfterPreProcessing.append(newLine)

print ("Befor steming"+ linesOfTestDataAfterPreProcessing[0])

linesOfTestDataAfterSteming= []
for line in linesOfTestDataAfterPreProcessing:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            print (w +" stemed to : "+st.stem(w))
            newLine = newLine.replace(w, st.stem(w)) # for identifting words
    linesOfTestDataAfterSteming.append(newLine)

print("After steming"+ linesOfTestDataAfterSteming[0])

wordsInTraining = {}
for d in linesOfTestDataAfterPreProcessing:
    for w in d.split():
        if w not in wordsInTraining:
            wordsInTraining[w] = 1
        else:
            wordsInTraining[w] += 1
print("Number of unique words in Test: %d." % len(wordsInTraining))


print ("First line after cleanup from test Data: : ", linesOfTestDataAfterSteming[0])

vectorizer.fit_transform(linesOfTestDataAfterPreProcessing)
print (vectorizer.vocabulary_)

#Sparce vectore for training
smatrixFromTesting = vectorizer.transform(linesOfTestDataAfterPreProcessing)
print(smatrixFromTesting)

print('-----')

smatrixFromTesting = cosine_similarity(smatrixFromTesting, dense_output=False)


#result = 1 - spatial.distance.cosine(smatrixFromTraining, smatrixFromTesting)

f = open('TestData/Test/format_out.dat', 'w')
for vt in smatrixFromTesting:
    cosineSimilarityValues=[]
    for vS in smatrixFromTraining:
        dotProduct = vt.T.dot(vS)[0,0]
        lengtht = np.linalg.norm(vt.data)
        lengthS = np.linalg.norm(vS.data)

        #handle exceptions
        cosineSimilarityValue= dotProduct/(lengtht*lengthS)
        cosineSimilarityValues.append(cosineSimilarityValue)

    kneighbours = heapq.nlargest(9, cosineSimilarityValues)

    neighbourReviewTypeList = []
    neighbourReviewTypeNegative = 0
    neighbourReviewTypePositive = 0

    for neighbour in kneighbours:
            index = cosineSimilarityValues.index(neighbour)

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






    print(len(cosineSimilarityValues));





print('-----')




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



