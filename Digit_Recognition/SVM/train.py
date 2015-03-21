# coding utf-8

from sklearn import svm
import numpy as np

from kernel import my_kernel

class Model:
    
    def __init__(self, model, training_size, clf, model_type):
        
        self.model = model
        self.training_size = training_size
        self.testing_size = clf.testing_size
        
        self.type = model_type
        self.X_train = clf.shuffledData[:training_size]
        self.y_train = clf.shuffledLabels[:training_size]
        self.X_test = clf.shuffledData[training_size:clf.testing_size+training_size]
        self.actual_Digits = clf.shuffledLabels[training_size:clf.testing_size+training_size]
        
        self.accuracy = 0
        self.error_rate = 1
        self.error_rate_on_training = 1
        self.error_indices = list()
        self.biased_accuracy = 0
        
    def train(self):
        
        if self.type=='linear':
            
            self.model.fit(self.X_train, self.y_train)
            
        else:
        
            self.mat_train = np.zeros((self.training_size,self.training_size))
            for i in range(self.training_size):
                for j in range(self.training_size):
                    self.mat_train[i][j]= my_kernel(self.X_train[i], self.X_train[j], 0)
            self.model.fit(self.mat_train, self.y_train)            
        
    def test(self):
        
        if self.type=='linear':
            self.predicted_Digits = self.model.predict(self.X_test)
            
        else:
            mat_test = np.zeros((self.testing_size, self.training_size))
        
            for i in range(self.testing_size):
                for j in range(self.training_size):
                    mat_test[i][j]= my_kernel(self.X_test[i], self.X_train[j], 0)
    
            self.predicted_Digits = self.model.predict(mat_test)
        
        error_index = 0   
        for elem1, elem2 in zip(self.predicted_Digits, self.actual_Digits):
            if elem1 == elem2:
                self.accuracy += 1
            else:
                self.error_indices.append(error_index)
            error_index += 1
                
        self.accuracy *= 100. / self.testing_size
        self.error_rate = 100 -self.accuracy
        
        print "Error Rate for " + self.type + " model : " + str(self.error_rate) + "%"
        print 50*'-'
                
    def test_on_training_data(self):
        
        if self.type=='linear':
            predicted_Digits = self.model.predict(self.X_train)
            
        else:
            predicted_Digits = self.model.predict(self.mat_train)
           
        for elem1, elem2 in zip(predicted_Digits, self.y_train):
            if elem1 == elem2:
                self.biased_accuracy += 1
                
        self.biased_accuracy *= 100. / self.training_size
        self.biased_error_rate = 100 -self.biased_accuracy
        
        print "Error Rate for " + self.type + " model on training data: " + str(self.biased_error_rate) + "%"
        print 50*'-'
        
        

def train_svm(clf): 
    """train data"""  
     
    ############# TRAIN SVM ############# 
    print 50*'='
    print "SVM TRAINING"
    print 50*'='
        
    cust_models = list()
    lin_models = list()
    
    for size in clf.training_sizes:
        
        print "Training with training " + str(size) + " samples"
        print 50*'='
        
        if clf.DEBUG:
            print 50*'-'
            print "Shuffled Data and Label shape: ", len(clf.shuffledData), len(clf.shuffledLabels)
        
        
        custom_model = svm.SVC(C=29.2, kernel='precomputed') 
        cust_models+=[Model(custom_model, size, clf, 'custom')]
        cust_models[-1].train()
        cust_models[-1].test()
        cust_models[-1].test_on_training_data()
        
        linear_model=svm.SVC(C=1.4, kernel='linear')
        lin_models+=[Model(linear_model, size, clf, 'linear')]
        lin_models[-1].train()
        lin_models[-1].test()
        lin_models[-1].test_on_training_data()
        
    clf.cust_models = cust_models
    clf.lin_models = lin_models
        
        
    
#     if clf.visualization:   
#         for error in error_indices:
#             plt.imshow(clf.imageComplete[error][0])
#             plt.show()    