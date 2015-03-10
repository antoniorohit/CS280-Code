%###################################
%# Function to calculate cross validation scores 
%# Input: SVC object, data, labels, num folds
%# Output: Array of scores averaged for each fold
%###################################
def computeCV_Score(gauss_params, data, labels, folds, weight):
    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    (all_mu, all_cov, all_prior) = gauss_params
    %# For each fold trained on...
    for i in range(folds):
        %# Initialize variables
        j = 0
        accuracy = 0
        for j in range(folds):
            if(j!=i):
                predicted_Digits = gauss_predict(data[j], all_mu, all_cov, all_prior, False, weight)
                for (elem1, elem2) in zip(predicted_Digits, labels[j]):
                    if elem1 == elem2:
                        accuracy+=1
                    else:
                        pass
#                         print data[i].shape
#                         plt.imshow(data[i])
#                         plt.show()
                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Digits)))
        i+=1
    return np.array(scores)
%###################################