def train_gauss(im_data, labels):
    %############# 
    %# Fit gaussian to each digit
    %############# 
    all_cov = []
    all_prior = []
    all_mu = []
        
    for i in range(0,2):
        data_label = []
        index  = 0
        for label in labels:
            if(label == i):
                data_label.append(im_data[index].flatten())
            index += 1
        %# data label contains all the data of a certain number
        %# transpose it so that we can take the average across 
        %# all data
        data_label = np.transpose(np.array(data_label))
        
        %# mean mu for label 
        mu = [np.mean(data_label[j]) for j in range(0, len(data_label))]
        
        %# sample covariance
        cov = np.zeros((len(data_label), len(data_label)))
        data_label = data_label.T 
        for elem in (data_label):
            cov += np.mat(elem-mu).T*np.mat(elem-mu)
        cov = cov/len(data_label)
    
    
        %# prior probability 
        prior = float(len(data_label))/(len(im_data))
            
#         print prior, i
        all_cov.append(cov)
        all_prior.append(prior)
        all_mu.append(mu)
    return all_mu, all_cov, all_prior
