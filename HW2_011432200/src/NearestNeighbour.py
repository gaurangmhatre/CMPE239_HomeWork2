#Imports for code
import numpy as np
import scipy as sp
#%matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend
import nltk

from nltk.corpus import stopwords

with open("TestData/train.dat", "r") as fh:
    linesOfTrainData = fh.readlines()
len(linesOfTrainData)

# transform docs into lists of words
# docsTrain = [l.split() for l in linesOfTrainData]
# print("Train data:")
# print("Number of docs: %d." % len(docsTrain))
# print("Number of words: %d." % np.sum([len(d) for d in docsTrain]))

# Format data : open docs file and read its lines
with open("TestData/format.dat", "r") as fh:
    linesOfFormat = fh.readlines()
len(linesOfFormat)

# transform docs into lists of words
# docsFormat = [l.split() for l in linesOfFormat]
# print("Format data:")
# print("Number of docs: %d." % len(docsFormat))
# print("Number of words: %d." % np.sum([len(d) for d in docsFormat]))

# Test Data : open docs file and read its lines
with open("TestData/test.dat", "r") as fh:
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

linesOfTrainDataAfterPreProcessing = []
print ("First line before cleanup: ",linesOfTrainData[0])

for line in linesOfTrainData:
    newLine = line
    for w in newLine.split():
        if w.lower() in stops:
            newLine = newLine.replace(' '+w+' ', ' ')#for identifting words
    linesOfTrainDataAfterPreProcessing.append(newLine)


print ("First line after cleanup: ", linesOfTrainDataAfterPreProcessing[0])

# get a frequency count for all words in the corpus
wordsInTrainingData = {}
for d in linesOfTrainDataAfterPreProcessing:
    for w in d.split():
        if w == '+1' or w == '-1':
            continue;
        if w not in wordsInTrainingData:
            wordsInTrainingData[w] = 1
        else:
            wordsInTrainingData[w] += 1
print("Number of unique words in training data: %d." % len(wordsInTrainingData))


# for review in linesOfTrainDataAfterPreProcessing
#     for w in review.split():
#         if w.strip()[0] == '-':
#
#         elif w.strip()[0] == '+':
#
#         break

