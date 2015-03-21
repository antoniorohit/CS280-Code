def gauss_predict(data, all_mu, all_cov, all_prior, overall = True, weight = 0.001):
    labelled_list = []
    
    %Create PDFs
    n = []
    for label in range(0, 2):
        mu = all_mu[label]
        if overall == True:
            cov = np.mean(all_cov)
        else:
            %# add small value to diag to remove singularity
            small_value = weight*np.eye(len(all_cov[0]))
            cov = all_cov[label] + small_value
        n.append(norm(mean=mu, cov=(cov)))
                
    for elem in data:
        prob_list = []
        for label in range(0, 2):
            prob_list.append(np.sum(n[label].logpdf(elem))*all_prior[label])
        labelled_list.append(prob_list.index(max(prob_list)))
    
    return labelled_list
