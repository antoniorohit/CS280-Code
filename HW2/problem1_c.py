# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage

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

def my_kernel(x, y):
    mat_train = np.zeros((np.shape(x)[0], np.shape(x)[0]))
    print mat_train.shape
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[0]):
            mat_train[i][j]= np.sum(np.minimum(x[i], y[j]))
    return mat_train

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

for i in range(len(imageData)):
    if np.linalg.norm(imageData[i]) != 0:
        imageData[i] = (imageData[i]/np.linalg.norm(imageData[i]))

imageComplete = zip(imageData, imageLabels)
    
############# SET ASIDE VALIDATION DATA (10,000) ############# 
# SHUFFLE THE IMAGES
random.shuffle(imageComplete)

# Arrays to hold the shuffled data and labels
shuffledData = []
shuffledLabels = []

############# 
# ORIENTATION HISTOGRAM
############# 
for elem in imageComplete:
    grad_filter = np.array([[-1.0, 0.0, 1.0]])
    gradx = signal.fftconvolve(elem[0], grad_filter, 'same')
    grady = signal.fftconvolve(elem[0], np.transpose(grad_filter), 'same')
    
    mag = np.sqrt(np.square(gradx) + np.square(grady))
    # SUPER COOL! View the Edges!
#         plt.imshow(np.absolute(mag), cmap='gray')   
#         plt.show()
    
    # NOTE! step size shift not implemented
    cell_size = 4
    cell_4 = (1.0/cell_size**2)*np.ones((cell_size,cell_size))
    cell_size = 7
    cell_7 = (1.0/cell_size**2)*np.ones((cell_size,cell_size))

     
    ori = np.array(np.arctan2(grady, gradx))
    # Aggregate over a cell 4x4 etc moved by half cell size
    ori_4 = signal.fftconvolve(ori, cell_4, 'valid').flatten()[::2]
    ori_7 = signal.fftconvolve(ori, cell_7, 'valid').flatten()[::3.5]

    # array, num bins, range
    ori_4_hist = np.histogram(ori_4, 8, (-np.pi, np.pi))[0]
    ori_7_hist = np.histogram(ori_7, 8, (-np.pi, np.pi))[0]
    
    shuffledData.append(np.append(2*ori_4_hist/np.linalg.norm(ori_4_hist), ori_7_hist/np.linalg.norm(ori_7_hist)))
    shuffledLabels.append((elem[1][0]))


############# TRAIN SVM ############# 
print 50*'='
print "SVM TRAINING"
print 50*'='

errorRate_array = []
C = np.linspace(1,3,16)                   # array of values for parameter C
training_Size = [100, 200, 500, 1000, 2000, 5000, 10000, 15000]
test_size = 10000
for elem in training_Size:
    if DEBUG:
        print 50*'-'
        print "Shuffled Data and Label shape: ", len(shuffledData), len(shuffledLabels)
    
    X_train = shuffledData[:elem]
    y_train = shuffledLabels[:elem]
    
    svc = svm.SVC(kernel='precomputed')
    mat_train = np.zeros((elem,elem))
    for i in range(elem):
        for j in range(elem):
            mat_train[i][j]= np.sum(np.minimum(X_train[i], X_train[j]))
    svc.fit(mat_train, y_train)
             
    X_test = shuffledData[50000:]
    mat_test = np.zeros((test_size,elem))
    
    for i in range(test_size):
        for j in range(elem):
            mat_test[i][j]= np.sum(np.minimum(X_train[j], X_test[i]))
    predicted_Digits = svc.predict(mat_train)
    
#     clf = svm.SVC(kernel=my_kernel, C=1.4)
#     print('Shuffled Data Shape:', np.shape(shuffledData[:elem]))
#     clf.fit(shuffledData[:elem], np.array(shuffledLabels[:elem]))
#     predicted_Digits = clf.predict(shuffledData[50000:])
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
plt.savefig("./Results/ErrorRate_TrainingSize_OHG_"  + ".png")

####################################### 

print 50*'='
print "End of File"
print 50*'='
