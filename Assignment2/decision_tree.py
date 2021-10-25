# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from subprocess import call

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    # INSERT YOUR CODE HERE
    partitions = {}
    index = 0
    for i in x:
        if i in partitions:
            partitions[i].append(index)
        else:
            partitions[i] = [index]
        index += 1
    return partitions
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    # INSERT YOUR CODE HERE
    entropy = 0
    total = 0
    for i in y.keys():
        total += len(y[i])
    for vi in y.keys():
        entropy += -(len(y[vi]) / total) * (np.log2(len(y[vi]) / total))
    return entropy
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):  # same as information gain
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    # INSERT YOUR CODE HERE
    I = entropy(partition(y))
    split = 0
    total = 0
    for i in partition(x).values():
        total += len(i)
    for p in partition(x).values():
        temp = [y[i] for i in p]
        split += (len(p) / total) * entropy(partition(temp))
    return I - split
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.
    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.
    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    a,b = np.unique(y,return_counts=True)

    nums = [-1, -1]
    for i in range(len(b)):
        if b[i] > nums[1]:
            nums=[i,b[i]]


    if list(y) == [y[0]] * len(y):
        return y[0]
    if depth == max_depth:
        return a[nums[0]]
    if attribute_value_pairs == None:
        return a[nums[0]]

    attributes = set()
    for val in attribute_value_pairs:
        attributes.add(val[0])
    max = -1

    for i in attributes:
        minfo = mutual_information(x[:, i], y)
        if max <= minfo:
            attr = i
            max = minfo

    part = partition(x[:, attr])
    gain = -1
    for val in attribute_value_pairs:
        if val[0] == attr:
            v = val[1]
            t = part[v]
            minf = mutual_information(x[t, attr], y)
            if minf > gain:
                value = v
                gain = minfo

    attribute_value_pairs.remove((attr, value))
    t = part[value]
    f = [i for i in range(len(x[:, 0])) if i not in t]

    try:
        return {
            (attr, value, False): id3(x[f], y[f], attribute_value_pairs, depth + 1, max_depth),
            (attr, value, True): id3(x[t], y[t], attribute_value_pairs, depth + 1, max_depth)}
    except:
        return a[nums[0]]
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.

    if type(tree)is not dict:
        return tree
    current = list(tree.keys())
    if x[current[0][0]] != current[0][1]:
        return predict_example(x, tree[current[0]])
    else:
        return predict_example(x, tree[current[1]])
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    # INSERT YOUR CODE HERE
    return (1 / len(y_true)) * sum(y_true != y_pred)
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/' #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    #default for autograder
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, avp, max_depth=3)
    
    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    """
    #part b
    #monks-1
    train_loss = []
    test_loss = []
    for depth in range(1, 11):
        M = np.genfromtxt('monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        M = np.genfromtxt('monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
        # set attribute value pairs
        avp = []
        for i in range(Xtrn.shape[1]):
            set_values = set()
            for j in Xtrn[:, i]:
                if j not in set_values:
                    set_values.add(j)
                    avp.append((i, j))
        decision_tree = id3(Xtrn, ytrn, avp, max_depth=depth)
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        trn_err = compute_error(ytrn, y_pred_trn)
        train_loss.append(trn_err)
        test_loss.append(tst_err)
    plt.figure(figsize=(10, 8))
    plt.plot([i for i in range(1, 11)], train_loss, label="train loss")
    plt.plot([i for i in range(1, 11)], test_loss, label="test loss")
    plt.title('Monk-1 Data')
    plt.xlabel('Tree depth', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    depths = [1,3,5]
    plt.xticks(list(depths), fontsize=12)
    plt.legend(['Train Error', 'Test Error'], fontsize=16, loc='upper right')
    # monks-2
    train_loss = []
    test_loss = []
    for depth in range(1, 11):
        M = np.genfromtxt('monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        M = np.genfromtxt('monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
        # set attribute value pairs
        avp = []
        for i in range(Xtrn.shape[1]):
            set_values = set()
            for j in Xtrn[:, i]:
                if j not in set_values:
                    set_values.add(j)
                    avp.append((i, j))
        decision_tree = id3(Xtrn, ytrn, avp, max_depth=depth)
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        trn_err = compute_error(ytrn, y_pred_trn)
        train_loss.append(trn_err)
        test_loss.append(tst_err)
    plt.figure(figsize=(10, 8))
    plt.plot([i for i in range(1, 11)], train_loss, label="train loss")
    plt.plot([i for i in range(1, 11)], test_loss, label="test loss")
    plt.title('Monk-2 Data')
    plt.xlabel('Tree depth', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    depths = [1, 3, 5]
    plt.xticks(list(depths), fontsize=12)
    plt.legend(['Train Error', 'Test Error'], fontsize=16, loc='upper right')
    #monks-3
    train_loss = []
    test_loss = []
    for depth in range(1, 11):
        M = np.genfromtxt('monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        M = np.genfromtxt('monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
        #set attribute value pairs
        avp = []
        for i in range(Xtrn.shape[1]):
            set_values = set()
            for j in Xtrn[:, i]:
                if j not in set_values:
                    set_values.add(j)
                    avp.append((i, j))
        decision_tree = id3(Xtrn, ytrn, avp, max_depth=depth)
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        trn_err = compute_error(ytrn, y_pred_trn)
        train_loss.append(trn_err)
        test_loss.append(tst_err)
    plt.figure(figsize=(10, 8))
    plt.plot([i for i in range(1, 11)], train_loss, label="train loss")
    plt.plot([i for i in range(1, 11)], test_loss, label="test loss")
    plt.title('Monk-3 Data')
    plt.xlabel('Tree depth', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    depths = [1, 3, 5]
    plt.xticks(list(depths), fontsize=12)
    plt.legend(['Train Error', 'Test Error'], fontsize=16, loc='upper right')
    plt.show()
    """
    """
    #part c
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    # Learn a decision tree of depth 1
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))
    decision_tree1 = id3(Xtrn, ytrn, avp, max_depth=1)
    # Pretty print it to console
    pretty_print(decision_tree1)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree1)
    render_dot_file(dot_str, './my_learned_tree1')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree1) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error of my tree = {0:4.2f}%.'.format(tst_err * 100))

    # matrix
    y_dt = np.array([predict_example(x, decision_tree1) for x in Xtst])
    print('Confusion matrix for depth {}'.format(1))
    print(confusion_matrix(ytst, y_dt))
    dot_str = to_graphviz(decision_tree1)
    render_dot_file(dot_str, './confusionmatrix'.format(1))

    # Learn a decision tree of depth 3
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))

    decision_tree3 = id3(Xtrn, ytrn, avp, max_depth=3)
    # Pretty print it to console
    pretty_print(decision_tree3)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree3)
    render_dot_file(dot_str, './my_learned_tree2')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree3) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error of my tree = {0:4.2f}%.'.format(tst_err * 100))
    # matrix
    y_dt = np.array([predict_example(x, decision_tree3) for x in Xtst])
    print('Confusion matrix for depth {}'.format(3))
    print(confusion_matrix(ytst, y_dt))
    dot_str = to_graphviz(decision_tree3)
    render_dot_file(dot_str, './confusionmatrix'.format(3))

    # Learn a decision tree of depth 5
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))
    decision_tree5 = id3(Xtrn, ytrn, avp, max_depth=5)

    # Pretty print it to console
    pretty_print(decision_tree5)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree5)
    render_dot_file(dot_str, './my_learned_tree3')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree5) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error of my tree = {0:4.2f}%.'.format(tst_err * 100))
    #matrix
    y_dt = np.array([predict_example(x, decision_tree5) for x in Xtst])
    print('Confusion matrix for depth {}'.format(5))
    print(confusion_matrix(ytst, y_dt))
    dot_str = to_graphviz(decision_tree5)
    render_dot_file(dot_str, './confusionmatrix'.format(5))
    """
    """
    #part d
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    # depth 1
    decision_tree_1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    decision_tree_1 = decision_tree_1.fit(Xtrn, ytrn)
    y_dt_1 = decision_tree_1.predict(Xtst)
    err_1 = compute_error(ytst, y_dt_1)
    confusion_matrix(ytst, y_dt_1)
    export_graphviz(decision_tree_1, out_file="myTree1.dot", filled=True, rounded=True)
    call(['dot', '-T', 'png', 'myTree1.dot', '-o', 'myTree1.png'])
    # depth 3
    decision_tree_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    decision_tree_3 = decision_tree_3.fit(Xtrn, ytrn)
    y_dt_3 = decision_tree_3.predict(Xtst)
    err_3 = compute_error(ytst, y_dt_3)
    confusion_matrix(ytst, y_dt_3)
    export_graphviz(decision_tree_3, out_file="myTree3.dot", filled=True, rounded=True)
    call(['dot', '-T', 'png', 'myTree3.dot', '-o', 'myTree3.png'])
    # depth 5
    decision_tree_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    decision_tree_5 = decision_tree_5.fit(Xtrn, ytrn)
    y_dt_5 = decision_tree_5.predict(Xtst)
    err_5 = compute_error(ytst, y_dt_5)
    confusion_matrix(ytst, y_dt_5)
    export_graphviz(decision_tree_5, out_file="myTree5.dot", filled=True, rounded=True)
    call(['dot', '-T', 'png', 'myTree5.dot', '-o', 'myTree5.png'])
    """
    """
    #part e
    #partc of e
    # Load the training data
    M = np.genfromtxt('./data.train.txt', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, -1]
    Xtrn = M[:,0:-1]

    # Load the test data
    M = np.genfromtxt('./data.test.txt', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, -1]
    Xtst = M[:,0:-1]
    # Learn a decision tree of depth 1
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))
    decision_tree1 = id3(Xtrn, ytrn, avp, max_depth=1)
    # Pretty print it to console
    pretty_print(decision_tree1)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree1)
    render_dot_file(dot_str, './my_sample_learned_tree1')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree1) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error of my tree = {0:4.2f}%.'.format(tst_err * 100))

    # matrix
    y_dt = np.array([predict_example(x, decision_tree1) for x in Xtst])
    print('Confusion matrix for depth {}'.format(1))
    print(confusion_matrix(ytst, y_dt))

    # Learn a decision tree of depth 3
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))

    decision_tree3 = id3(Xtrn, ytrn, avp, max_depth=3)
    # Pretty print it to console
    pretty_print(decision_tree3)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree3)
    render_dot_file(dot_str, './my_sample_learned_tree2')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree3) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error of my tree = {0:4.2f}%.'.format(tst_err * 100))
    # matrix
    y_dt = np.array([predict_example(x, decision_tree3) for x in Xtst])
    print('Confusion matrix for depth {}'.format(3))
    print(confusion_matrix(ytst, y_dt))


    # Learn a decision tree of depth 5
    # set attribute value pairs
    avp = []
    for i in range(Xtrn.shape[1]):
        set_values = set()
        for j in Xtrn[:, i]:
            if j not in set_values:
                set_values.add(j)
                avp.append((i, j))
    decision_tree5 = id3(Xtrn, ytrn, avp, max_depth=5)

    # Pretty print it to console
    pretty_print(decision_tree5)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree5)
    render_dot_file(dot_str, './my_sample_learned_tree3')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree5) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error of my tree = {0:4.2f}%.'.format(tst_err * 100))
    # matrix
    y_dt = np.array([predict_example(x, decision_tree5) for x in Xtst])
    print('Confusion matrix for depth {}'.format(5))
    print(confusion_matrix(ytst, y_dt))


    # part d of e
    # Load the training data
    M = np.genfromtxt('./data.train.txt', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data.test.txt', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    # depth 1
    decision_tree_1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    decision_tree_1 = decision_tree_1.fit(Xtrn, ytrn)
    y_dt_1 = decision_tree_1.predict(Xtst)
    err_1 = compute_error(ytst, y_dt_1)
    confusion_matrix(ytst, y_dt_1)
    export_graphviz(decision_tree_1, out_file="mySampleTree1.dot", filled=True, rounded=True)
    call(['dot', '-T', 'png', 'mySampleTree1.dot', '-o', 'mySampleTree1.png'])
    # depth 3
    decision_tree_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    decision_tree_3 = decision_tree_3.fit(Xtrn, ytrn)
    y_dt_3 = decision_tree_3.predict(Xtst)
    err_3 = compute_error(ytst, y_dt_3)
    confusion_matrix(ytst, y_dt_3)
    export_graphviz(decision_tree_3, out_file="mySampleTree3.dot", filled=True, rounded=True)
    call(['dot', '-T', 'png', 'mySampleTree3.dot', '-o', 'mySampleTree3.png'])
    # depth 5
    decision_tree_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    decision_tree_5 = decision_tree_5.fit(Xtrn, ytrn)
    y_dt_5 = decision_tree_5.predict(Xtst)
    err_5 = compute_error(ytst, y_dt_5)
    confusion_matrix(ytst, y_dt_5)
    export_graphviz(decision_tree_5, out_file="mySampleTree5.dot", filled=True, rounded=True)
    call(['dot', '-T', 'png', 'mySampleTree5.dot', '-o', 'mySampleTree5.png'])
    """


