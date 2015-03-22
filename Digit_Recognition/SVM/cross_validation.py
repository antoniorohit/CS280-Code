import numpy as np
from sklearn import svm
 

def KFoldCV(classifier, data, labels, folds):
    """compute k-fold cross-valida,tion score
    
    :param clf (scikit-learn classifier) classifier to evaluate
    :param data (list) dataset
    :param labels (list) labels of dataset
    :param folds (int) k of k-fold cross-validation
    :return: scores (numpy array) list of k scores"""
    
    i,j = 0,0
    scores = list()

    # For each fold trained on...
    for i in range(folds):
        # Initialize variables
        clf_local = classifier
        j = 0
        accuracy = 0

        clf_local.fit(data[i], labels[i])
        # For each validation performed (k-1 total) on a fold
        for j in range(folds):
            if(j!=i):
                predicted_Digits = clf_local.predict(data[j])
                for (elem1, elem2) in zip(predicted_Digits, labels[j]):
                    if elem1 == elem2:
                        accuracy+=1
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Digits)))
        i+=1
    return np.array(scores)


def crossValidation(clf):
    """compute cross validation and print results.
    
    :param suffledData : dataset with suffled indices
    :param shuffledLabels : corresponding labels with shuffled indices"""
    
    print 50*'='
    print "CROSS VALIDATION"
    print 50*'='
    
    ############# DATA PARTIONING ############# 
    crossValidation_Data= []
    crossValidation_Labels = []
    k = 10 
    lengthData = 1000
    stepLength = k
    C = np.linspace(1,3,16) 
    for index in range(0,k):
        crossValidation_Data.append(clf.shuffledData[index:lengthData:stepLength])
        crossValidation_Labels.append(clf.shuffledLabels[index:lengthData:stepLength])
    
    if clf.DEBUG:
        print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
        print 50*'-'
    
    scoreBuffer = []
    
    ############# CROSS-VALIDATION ############# 
    for C_Value in C:
        classifier = svm.SVC(kernel='linear', C=C_Value)
        scores = KFoldCV(classifier, crossValidation_Data, crossValidation_Labels, k)
        scoreBuffer.append((scores).mean())
        if clf.DEBUG:
            print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
            print 50*'-'
    
    # keep maximal score
    maxScore = np.max(np.array(scoreBuffer))
    maxScore_Index = scoreBuffer.index(maxScore)
    
    # Train SVM using best C value
    classifier = svm.SVC(kernel='linear', C=C[maxScore_Index])
    classifier.fit(clf.shuffledData[:10000], np.array(clf.shuffledLabels[:10000]))
    
    # Predict digits
    predicted_Digits = classifier.predict(clf.shuffledData[50000:])
    actual_Digits = clf.shuffledLabels[50000:]
    
    # Compute Accuracy
    accuracy = 0.0
    for elem1, elem2 in zip(predicted_Digits, actual_Digits):
        if elem1 == elem2:
            accuracy+=1
    
    # tell user best accuracy obtained
    print "Using Custom CV Function"
    print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", (100.0*accuracy/len(predicted_Digits))
    print 50*'-'

    print 50*'='
    print "End of File"
    print 50*'='