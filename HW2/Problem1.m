DEBUG = False
%############# FILE STUFF ############# 
File_Spam = './spam-dataset/spam_data.mat'


trainMatrix = io.loadmat(File_Spam)                 % Dictionary


%############# GET DATA ############# 
trainingData = np.array(trainMatrix['training_data'])
trainingLabels = np.array(trainMatrix['training_labels'][0])
testData= np.array(trainMatrix['test_data'])

%############# 
%# Normalization
%############# 
i = 0
for element in trainingData:
    if np.linalg.norm(trainingData[i]) != 0:
        trainingData[i] = (trainingData[i]/np.linalg.norm(trainingData[i]))
    i+=1

trainingComplete = zip(trainingData, trainingLabels)

%############# SHUFFLE DATA ############# 
random.shuffle(trainingComplete)
shuffledData = []
shuffledLabels = []
for elem in trainingComplete:
    shuffledData.append((elem[0]))                % Use a simple array as the feature
    shuffledLabels.append((elem[1]))

trainingData = np.array(shuffledData)
trainingLabels = np.array(shuffledLabels)
        

%#########################################################
%# CROSS VALIDATION 
%#########################################################
print 50*'='
print 'CROSS VALIDATION'
print 50*'='

%############# DATA PARTIONING ############# 
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(trainingData[index:-1:stepLength])
    crossValidation_Labels.append(trainingLabels[index:-1:stepLength])

scoreBuffer = []

%############# CROSS-VALIDATION ############# 
gauss_params = train_gauss(shuffledData[0:-1], shuffledLabels[0:-1])
small_weight = np.linspace(0.001, 0.00001, 5)
for weight in small_weight:
    scores = computeCV_Score(gauss_params, crossValidation_Data, crossValidation_Labels, k, weight)
    scoreBuffer.append((scores).mean())
    if 1:
        print 'Weight:', weight, 'Accuracy: %0.2f (+/- %0.2f)' % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'

maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print 'Best weight Value:', small_weight[maxScore_Index], 'Accuracy for that:', maxScore
print 50*'-'

