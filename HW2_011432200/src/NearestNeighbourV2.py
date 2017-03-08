import numpy as np

import scipy as sp
from scipy import spatial

import matplotlib.pyplot as plt # side-stepping mpl backend

from nltk.stem.lancaster import LancasterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import cross_validation
import heapq

vectorizer = CountVectorizer()
st = LancasterStemmer()


with open("TestData/train.dat", "r") as fh:
    #with open("TestData/Test/training_out.dat", "r") as fh:
    linesOfTrainData = fh.readlines()
len(linesOfTrainData)

#with open("TestData/format.dat", "r") as fh:
with open("TestData/Test/format_out.dat", "r") as fh:
    linesOfFormat = fh.readlines()
len(linesOfFormat)

with open("TestData/test.dat", "r") as fh:
#with open("TestData/Test/test_out.dat", "r") as fh:
    linesOfTestData = fh.readlines()
len(linesOfTestData)

print ("First line after cleanup from test Data: ",linesOfTrainData[0])

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(linesOfTrainData, linesOfTestData, test_size= 0.1,train_size= 0.9, random_state=42)


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
linesOfTrainData_Transformed =  vectorizer.fit_transform(features_train)
linesOfTestData_Transformed = vectorizer.transform(features_test)


selector = SelectPercentile(f_classif, percentile=10)
selector.fit(linesOfTrainData_Transformed, labels_train) #labels_train
linesOfTrainData_Transformed = selector.transform(linesOfTrainData_Transformed)
linesOfTestData_Transformed = selector.transform(linesOfTestData_Transformed)


f = open('TestData/Test/format_out.dat', 'w')
for vt in linesOfTestData_Transformed:
    cosineSimilarityValues=[]
    for vS in linesOfTrainData_Transformed:
        dotProduct = vt.dot(np.transpose(vS))
        lengtht = np.linalg.norm(vt.data)
        lengthS = np.linalg.norm(vS.data)

        #handle exceptions

        if lengthS!=0 and lengtht!=0 :
            cosineSimilarityValue= dotProduct/(lengtht*lengthS)
        else:
            cosineSimilarityValue= 0
        cosineSimilarityValues.append(cosineSimilarityValue)

    kneighbours = heapq.nlargest(3, cosineSimilarityValues)
    #kneighbours = sp.sparse.csr_matrix.max(cosineSimilarityValues)

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

print("-----")