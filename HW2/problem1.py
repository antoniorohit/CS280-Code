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

DEBUG = False
############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"


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

############# 
# Ink Normalization
############# 
i = 0
#TODO change for loop
for element in imageData:
    if np.linalg.norm(imageData[i]) != 0:
        imageData[i] = (imageData[i]/np.linalg.norm(imageData[i]))
    i+=1

imageComplete = zip(imageData, imageLabels)

############# SET ASIDE VALIDATION DATA (10,000) ############# 
# SHUFFLE THE IMAGES
random.shuffle(imageComplete)

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
    
for MALIK in [0]:#, 2, 1]:
    print "Feature Iteration:", MALIK
    
    # Arrays to hold the shuffled data and labels
    shuffledData = []
    shuffledLabels = []
    
    index = 0
    
    if MALIK == 0:      # Orientation histogram + magnitude of gradient
        for elem in imageComplete:
            if index%100 == 0:
                print index
            index +=1
            grad_filter = np.array([[-1, 0, 1]])
            gradx = signal.convolve2d(elem[0], grad_filter, 'same')
            grady = signal.convolve2d(elem[0], np.transpose(grad_filter), 'same')
            
            ori = np.array(np.arctan2(grady, gradx))
            mag = np.sqrt(np.square(gradx) + np.square(grady))
            
            # SUPER COOL! View the Edges!
#             plt.figure()
#             plt.imshow((mag), cmap='gray')   
#             plt.figure()
#             plt.imshow(np.absolute(ori), cmap='gray')   
#             plt.show()
            
            ori_4_hist = []
            ori_7_hist = []
            
#             
            ori_4_1 = blockshaped(ori, 4, 4)
            ori_4_2 = blockshaped(ori[2:-2, 2:-2], 4, 4)
            for (elem1, elem2) in zip(ori_4_1, ori_4_2):
                ori_4_hist.append(np.histogram(elem1.flatten(), 9, (-np.pi, np.pi))[0])
                ori_4_hist.append(np.histogram(elem2.flatten(), 9, (-np.pi, np.pi))[0])
 
            ori_7_1 = (blockshaped(ori, 7, 7))
            ori_7_2 = (blockshaped(ori[3:-4, 3:-4], 7, 7))
            for elem1, elem2 in zip(ori_7_1, ori_7_2):
                ori_4_hist.append(np.histogram(elem1.flatten(), 9, (-np.pi, np.pi))[0])
                ori_4_hist.append(np.histogram(elem2.flatten(), 9, (-np.pi, np.pi))[0])

#              
#             for i in np.linspace(0, 28, num=15):
#                 for j in np.linspace(0, 28, num=15):
#                     if(i%2 == 0 and j%2 == 0 and i<=28-4 and j<=28-4):
#                         ori_4 = ori[i:i+4, j:j+4].flatten()
# #                         print np.shape(ori_4), i, j
#                         ori_4_hist.append(np.histogram(ori_4, 10, (-np.pi, np.pi))[0])
#   
#             for i in np.linspace(0, 28, num=8):
#                 for j in np.linspace(0, 28, num=8):
#                     if(i%4 == 0 and j%4 == 0 and i<=28-7 and j<=28-7):
#                         ori_7 = ori[i:i+7, j:j+7].flatten()
# #                         print np.shape(ori_7), i, j
#                         ori_7_hist.append(np.histogram(ori_7, 10, (-np.pi, np.pi))[0])
                        
            
            ori_4_hist = np.float64(ori_4_hist)/(np.linalg.norm(ori_4_hist))
            ori_7_hist = np.float64(ori_7_hist)/(np.linalg.norm(ori_7_hist))
            
            
#             print ori_4_hist, ori_7_hist
            
#             plt.figure()
#             plt.hist(ori_4_hist)
#             plt.show()
            
            shuffledData.append(np.append(ori_4_hist, ori_7_hist))
            shuffledLabels.append((elem[1][0]))
        
    elif MALIK == 1:        # extra features of smoothed image
        for elem in imageComplete:
            cell_size = 4
            cell = (1.0/cell_size**2)*np.ones((cell_size,cell_size))
            smooth_4 = signal.fftconvolve(elem[0], cell, 'valid')
            smooth_4 = smooth_4.flatten()[::cell_size/2]                      #pick every second element
            cell_size = 7
            cell = (1.0/cell_size**2)*np.ones((cell_size,cell_size))
            smooth_7 = signal.fftconvolve(elem[0], cell, 'valid')
            smooth_7 = smooth_7.flatten()[::cell_size/2]                      #pick every second element
            smooth_all = np.append(smooth_4, smooth_7)
            shuffledData.append(np.append(elem[0], smooth_all))
            shuffledLabels.append((elem[1][0]))
    else:           # raw pixels
        for elem in imageComplete:
            shuffledData.append((elem[0]).flatten())                # Use a simple array of pixels as the feature
            shuffledLabels.append((elem[1][0]))
        
    # NOTE: Set aside 50,000-60,000 to validate

    ############# TRAIN SVM ############# 
    print 50*'='
    print "SVM TRAINING"
    print 50*'='
    
    errorRate_array = []
    C = np.linspace(1,3,16)                   # array of values for parameter C
    training_Size = [100, 200, 500, 1000, 2000, 5000, 10000]
    for elem in training_Size:                
        clf = svm.SVC(kernel='linear', C=1.4)
        clf.fit(shuffledData[:elem], np.array(shuffledLabels[:elem]))
        
        predicted_Digits = clf.predict(shuffledData[50000:])
        actual_Digits = shuffledLabels[50000:]
        accuracy = 0.0
        for elem1, elem2 in zip(predicted_Digits, actual_Digits):
            if elem1 == elem2:
                accuracy+=1
        
        errorRate_array.append(100-100.0*accuracy/len(predicted_Digits))
        print "Training Size:", elem 
        print "Error Rate: ", errorRate_array[-1], "%"
        print 50*'-'
    
    
    # Plot error rate vs training size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Error Rate Vs Training Size')
    ax.set_ylabel('Training Size')
    ax.set_xlabel('Error Rate')
    ax.plot(training_Size, errorRate_array)
    for xy in zip(training_Size, errorRate_array):                                                # <--
        ax.annotate('%s' % int(xy[1]) + "%", xy=xy, fontsize = 'small') # <--
    plt.grid()
    plt.savefig("./Results/ErrorRate_TrainingSize_" + str(MALIK) + ".png")

####################################### 

#########################################################
# CROSS VALIDATION 
#########################################################
print 50*'='
print "CROSS VALIDATION"
print 50*'='

############# DATA PARTIONING ############# 
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
lengthData = 10000
stepLength = k
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

############# USING BUILT IN FUNCTION ############# 
# for C_Value in C:
#     clf = svm.SVC(kernel='linear', C=C_Value)
#     scores = cross_validation.cross_val_score(clf, shuffledData[:10000], shuffledLabels[:10000], cv=10)
#     if DEBUG:
#         print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
#         print 50*'-'
# 
# maxScore = scoreBuffer.max()
# maxScore_Index = scoreBuffer.index(maxScore)
# print "Using BuiltIn CV Function"
# print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", maxScore
# print 50*'-'


print 50*'='
print "End of File"
print 50*'='
