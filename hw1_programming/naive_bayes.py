"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np

def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """

    labels = np.unique(train_labels)

    d, n = train_data.shape #d is columns and n is rows
    num_classes = labels.size

    model = {}
    model['p(y)'] = np.zeros(num_classes)
    model['p(x|y)'] = np.zeros((num_classes, d))

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES (USING LAPLACE ESTIMATE)
    """
    Pr(X = True|Y = y) = [(# examples of class y where X = True) + 1 ] / [(Total # of examples of class y) + 2]
    """
    #p(y) calculations

    for i in range(num_classes):
        model['p(y)'][i] = np.sum(train_labels == labels[i]) / n
        split_matrix = train_data[:, labels[i] == train_labels]
        sums = np.sum(split_matrix, 1) # sums the axis 1 of the split_matrix hence the attr values in each row; returns ndarray

        #input d-shape array
        model['p(x|y)'][i, :] = (np.ravel(sums) + 1) / (split_matrix.shape[0] + 2)

    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA

    d, n = data.shape #features = d / examples = n ; #row/#col
    num_classes = model['p(y)'].size
    ddata = data
    idata = 1 - ddata #inverted probability
    fmodel = 1 - model['p(x|y)'] #inverted cond probability

    predicted_data = np.zeros((num_classes,n))

    for i in range(num_classes):
        
        prior = model['p(y)'][i]
        cond = model['p(x|y)'][i, :] #length d vector
       
        fcond = fmodel[i, :] #false conditional prob
        
        # Error bound checking for np.log for a zero argument to avoid the Runtime Warning#######
        logPrior = 0
        if (prior <= 0):
            logPrior = -999999999
        else:
            logPrior = np.log(prior)

        logCond = np.zeros(len(cond))
        flogCond = np.zeros(len(fcond))
        for j in range(cond.shape[0]):
            if cond[j] <= 0:
                logCond[j] = -999999999
            else:
                logCond[j] = np.log(cond[j])
                flogCond[j] = np.log(fcond[j])
        #########################################################################################
        
        # model class by feat and data is feat by examples
        # dot product cancels out feat so you are given class by example

        #i by exmaple aka 1 by n
        sum_scores = logCond.dot(ddata) + flogCond.dot(idata) + logPrior 

        predicted_data[i, :] = sum_scores

    max_scores = predicted_data.argmax(0) #max within each col, where each col is an example; # 1 x n
    return max_scores

