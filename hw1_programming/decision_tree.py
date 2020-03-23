"""This module includes methods for training and predicting using decision trees."""
import numpy as np


def calculate_information_gain(data, labels):
    """
    Computes the information gain for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape

    parent_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            parent_entropy -= class_prob * np.log(class_prob)

    # print("Parent entropy is %d\n" % parent_entropy)

    gain = parent_entropy * np.ones(d) #initialization of gains for every attribute

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n)) # number of examples containing each feature
    prob_x = num_x / n # fraction of examples containing each feature
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8) # probability of observing class c for each feature
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    node = dict()
    # TODO: INSERT YOUR CODE FOR LEARNING THE DECISION TREE STRUCTURE HERE
    if (depth >= max_depth or np.unique(labels).size == 1):
        u, c = np.unique(labels, return_counts=True)
        if len(u) == 0:
            node['prediction'] = 0
        else:
            node['prediction'] = int(u[c.argmax()])
        return node


    gain = calculate_information_gain(data, labels)
    gain_index = int(np.argmax(gain)) #obtain the index of the max gain

    node['gain'] = gain_index

    #non returns indexes where it is non-zero

    l_i = np.nonzero(1-data[gain_index])
    r_i = np.nonzero(data[gain_index])
    left_data = np.squeeze(data[:, l_i]) #all columns that contain a '0' within the row of the max_gain_index
    right_data = np.squeeze(data[:,r_i]) #all columns that contain a '1'

    node['left'] = recursive_tree_train(left_data, labels[l_i], depth+1, max_depth, num_classes)
    node['right'] = recursive_tree_train(right_data, labels[r_i], depth+1, max_depth, num_classes)

    return node


def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE FOR COMPUTING THE DECISION TREE PREDICTIONS HERE
    ddata = np.transpose(data)
    d,n = ddata .shape
    labels = np.zeros(d)
    lc = 0
    for i in ddata:
        cLabel = decision_tree_predict_helper(i, model)
        labels[lc] = cLabel
        lc+=1
    
    return labels


def decision_tree_predict_helper(rdata, model):
    #helper function
    if 'prediction' in model:
        return int(model['prediction'])
    else:
        # given a row of data, check gain index and see if '0' or '1'
        # if '0' send in recursive to left, else send in recursive to right
        gain_index = int(model['gain'])
        if rdata[gain_index] == 0:
            return decision_tree_predict_helper(rdata, model['left'])
        else:
            return decision_tree_predict_helper(rdata, model['right']) 
