# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage

# global bins 

###################################
# Function to calculate cross validation scores 
# Input: SVC object, data, labels, num folds
# Output: Array of scores averaged for each fold
###################################
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
def computeCV_Score(clf, data, labels, folds):
    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    clf_local = clf
    # For each fold trained on...
    for i in range(folds):
        # Initialize variables
        clf_local = clf
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
                    else:
                        pass
                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Digits)))
        i+=1
    return np.array(scores)
###################################

############# IMPORTS ############# 
from scipy import signal
import numpy as np
from sklearn import svm
from scipy import io
import random
import pylab as plt
from sklearn import cross_validation
import cPickle as pickle
import os
import scipy.ndimage.filters as filters

DEBUG = False
############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"

def my_kernel(x, y, first_dim):
    
    hist = np.sum(np.absolute(np.minimum(x[first_dim:],y[first_dim:])))
    lin = np.dot(x[:first_dim], y[:first_dim].T)
    return hist + lin
    
trainMatrix = io.loadmat(trainFileMNIST)                 # Dictionary
testMatrix = io.loadmat(testFileMNIST)                   # Dictionary

if DEBUG:
    print 50*'-'
    print trainMatrix, testMatrix

############# GET DATA ############# 
testData = np.array(testMatrix['test_images'])
testData = np.rollaxis(testData, 2, 0)                # move the index axis to be the first 
testData_flat = []
for elem in testData:
    testData_flat.append(elem.flatten())
imageData = np.array(trainMatrix['train_images'])
imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
imageLabels = np.array(trainMatrix['train_labels'])
gauss_bool = False
visualization = True

def getDataPickle(imageData, imageLabels):
    # Arrays to hold the shuffled data and labels
    shuffledData = list()
    shuffledLabels = list()

    if(os.path.isfile("./Results/shuffledData.p")):
        print('opening data with pickle...')
        shuffledData = pickle.load(open("./Results/shuffledData.p", 'rb'))
        print('data successfully opened')
        print('opening labels with pickle..')
        shuffledLabels = pickle.load(open("./Results/shuffledLabels.p", 'rb'))
        print('labels successfully opened')
    else:
        print "ERROR PICKLE FILE NOT FOUND"
        shuffledData, shuffledLabels, imageComplete = getDataMALIK(False, imageData, imageLabels)
        
    return shuffledData, shuffledLabels

def getDataMALIK(gauss_bool, imageData, imageLabels):
    # Arrays to hold the shuffled data and labels
    shuffledData = []
    shuffledLabels = []

    ############# 
    # Ink Normalization
    ############# 
    for i in range(len(imageData)):
        aux_norm = np.linalg.norm(imageData[i])
        if aux_norm != 0:
            imageData[i] /= aux_norm
    
    imageComplete = zip(imageData, imageLabels)
    
    if DEBUG:
        print 50*'-'
        print ("Shapes of image data and labels: ", imageData.shape, 
                                        imageLabels.shape, len(imageComplete))
        
        print "Image/Digit 10000:\n", imageComplete[20000]
        
    ############# SET ASIDE VALIDATION DATA (10,000) ############# 
    # SHUFFLE THE IMAGES
    random.shuffle(imageComplete)
    
    n_bins=9
    for ind in range(len(imageComplete)):
        if ind % 100 ==0:
            print 'feature extraction :' + str(np.around(ind*100./len(imageComplete), 2))+ ' % over'
        
        if gauss_bool:
            gaussFirst_x = filters.gaussian_filter1d(imageComplete[i][0], 1, order = 1, axis = 0)
            gaussFirst_y = filters.gaussian_filter1d(imageComplete[i][0], 1, order = 1, axis = 1)
            ori = np.array(np.arctan2(gaussFirst_y, gaussFirst_x))

        else:
            grad_filter = np.array([[-1, 0, 1]])
            gradx = signal.convolve2d(imageComplete[i][0], grad_filter, 'same')
            grady = signal.convolve2d(imageComplete[i][0], np.transpose(grad_filter), 'same')
            ori = np.array(np.arctan2(grady, gradx))
        
        ori_4_hist = list()
        ori_7_hist = list()
                     
        ori_4_1 = blockshaped(ori, 4, 4)
        ori_4_2 = blockshaped(ori[2:-2, 2:-2], 4, 4)
        for (elem1, elem2) in zip(ori_4_1, ori_4_2):
            ori_4_hist.append(np.histogram(elem1.flatten(), n_bins, (-np.pi, np.pi))[0])
            ori_4_hist.append(np.histogram(elem2.flatten(), n_bins, (-np.pi, np.pi))[0])
    
        ori_7_1 = (blockshaped(ori, 7, 7))
        ori_7_2 = (blockshaped(ori[3:-4, 3:-4], 7, 7))
        for elem1, elem2 in zip(ori_7_1, ori_7_2):
            ori_4_hist.append(np.histogram(elem1.flatten(), n_bins, (-np.pi, np.pi))[0])
            ori_4_hist.append(np.histogram(elem2.flatten(), n_bins, (-np.pi, np.pi))[0])
        
        ori_4_hist = np.float64(ori_4_hist)/(np.linalg.norm(ori_4_hist))
        ori_7_hist = np.float64(ori_7_hist)/(np.linalg.norm(ori_7_hist))
        
        shuffledData.append(np.append(ori_4_hist, ori_7_hist))
        shuffledLabels.append((imageComplete[i][1][0]))
                
    pickle.dump(shuffledData, open("./Results/shuffledData.p", 'wb'))
    pickle.dump(shuffledLabels, open("./Results/shuffledLabels.p", 'wb'))
        
    return shuffledData, shuffledLabels, imageComplete
    
def getDataNonMalik(imageComplete):
        # SHUFFLE THE IMAGES
    random.shuffle(imageComplete)
    
    # Arrays to hold the shuffled data and labels
    shuffledData = []
    shuffledLabels = []
    for elem in imageComplete:
        shuffledData.append((elem[0]).flatten())                # Use a simple array of pixels as the feature
        shuffledLabels.append((elem[1][0]))
    return shuffledData, shuffledLabels, imageComplete
        

def train_svm(shuffledData, shuffledLabels, visualization, imageComplete):    
    ############# TRAIN SVM ############# 
    print 50*'='
    print "SVM TRAINING"
    print 50*'='
    
    errorRate_array = []
    errorRate_array_on_training = []
    linear_errorRate_array = []
    linear_errorRate_array_on_training = []
    training_Size = [100, 200, 500, 1000] #, 2000, 5000, 10000]
    test_size = 10000
    for elem in training_Size:
        ind=0
        error_indices = list()
        if DEBUG:
            print 50*'-'
            print "Shuffled Data and Label shape: ", len(shuffledData), len(shuffledLabels)
        
        X_train = shuffledData[:elem]
        y_train = shuffledLabels[:elem]
        print X_train, elem
        svc = svm.SVC(C=29.2, kernel='precomputed')
        mat_train = np.zeros((elem,elem))
        for i in range(elem):
            for j in range(elem):
                print i, j
                mat_train[i][j]= my_kernel(X_train[i], X_train[j], 0)
        svc.fit(mat_train, y_train)
                
        X_test = shuffledData[elem:test_size+elem]
        mat_test = np.zeros((test_size,elem))

        for i in range(test_size):
            for j in range(elem):
                mat_test[i][j]= my_kernel(X_test[i], X_train[j], 0)
        
        predicted_Digits = svc.predict(mat_test)
        actual_Digits = shuffledLabels[elem:test_size+elem]

        accuracy = 0.0
        for elem1, elem2 in zip(predicted_Digits, actual_Digits):
            ind+=1
            if elem1 == elem2:
                accuracy+=1
            else:
                error_indices.append(ind)
        
        errorRate_array.append(100-100.0*accuracy/len(predicted_Digits))
        print "Training Size:", elem 
        print "Error Rate for custom model: ", errorRate_array[-1], "%"
        
        predictions=svc.predict(mat_train)
        
        accuracy = 0.0
        for elem1, elem2 in zip(predictions, y_train):
            if elem1 == elem2:
                accuracy+=1
        
        errorRate_array_on_training.append(100-100.0*accuracy/len(predictions))
        print "Error Rate for custom model on training set : ", errorRate_array_on_training[-1], "%"
        
        linear_svc=svm.SVC(C=1.4, kernel='linear')
        linear_svc.fit(X_train, y_train)
        predictions=linear_svc.predict(X_test)
        
        accuracy = 0.0
        for elem1, elem2 in zip(predictions, actual_Digits):
            if elem1 == elem2:
                accuracy+=1
        
        linear_errorRate_array.append(100-100.0*accuracy/len(predictions))
        print "Error Rate for linear model: ", linear_errorRate_array[-1], "%"
        
        predictions=linear_svc.predict(X_train)
        
        accuracy = 0.0
        for elem1, elem2 in zip(predictions, y_train):
            if elem1 == elem2:
                accuracy+=1
        
        linear_errorRate_array_on_training.append(100-100.0*accuracy/len(predictions))
        print "Error Rate for linear model on training set : ", linear_errorRate_array_on_training[-1], "%"
        print 50*'-'
    
    if visualization:   
        for error in error_indices:
            plt.imshow(imageComplete[error][0])
            plt.show()
        
    # Plot error rate vs training size
    
def plot_error_rate(training_Size, errorRate_array, linear_errorRate_array, label=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Error Rate Vs Training Size')
    ax.set_ylabel('Training Size')
    ax.set_xlabel('Error Rate')
    intersect, = ax.plot(training_Size, errorRate_array, label='Intersection kernel')
    linear, = ax.plot(training_Size, linear_errorRate_array, label='Linear kernel')
    ax.legend([intersect, linear], ['Intersection kernel', 'Linear kernel'])
    for xy in zip(training_Size, errorRate_array):                                                # <--
        ax.annotate('%s' % int(xy[1]) + "%", xy=xy, fontsize = 'small') # <--
    plt.grid()
    plt.savefig("./Results/ErrorRate" + label + ".png")
    plt.close()
    return

####################################### 

#########################################################
# CROSS VALIDATION 
#########################################################

def crossValidation(shuffledData, shuffledLabels):
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
        crossValidation_Data.append(shuffledData[index:lengthData:stepLength])
        crossValidation_Labels.append(shuffledLabels[index:lengthData:stepLength])
    
    if DEBUG:
        print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
        print 50*'-'
    
    scoreBuffer = []
    
    ############# CROSS-VALIDATION ############# 
    for C_Value in C:
        clf = svm.SVC(kernel='linear', C=C_Value)
        scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
        scoreBuffer.append((scores).mean())
        if DEBUG:
            print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
            print 50*'-'
    
    maxScore = np.max(np.array(scoreBuffer))
    maxScore_Index = scoreBuffer.index(maxScore)
    
    # Train SVM using best C value
    clf = svm.SVC(kernel='linear', C=C[maxScore_Index])
    clf.fit(shuffledData[:10000], np.array(shuffledLabels[:10000]))
    # Predict digits
    predicted_Digits = clf.predict(shuffledData[50000:])
    actual_Digits = shuffledLabels[50000:]
    # Compute Accuracy
    accuracy = 0.0
    for elem1, elem2 in zip(predicted_Digits, actual_Digits):
        if elem1 == elem2:
            accuracy+=1
    
    print "Using Custom CV Function"
    print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", (100.0*accuracy/len(predicted_Digits))
    print 50*'-'

    print 50*'='
    print "End of File"
    print 50*'='

imageComplete = zip(imageData, imageLabels)
shuffledData, shuffledLabels = getDataPickle(imageData, imageLabels)
train_svm(shuffledData, shuffledLabels, visualization, imageComplete)