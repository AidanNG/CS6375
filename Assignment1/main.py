def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
    return y

import numpy as np # For all our math needs
n = 750 # Number of data points
X = np.random.uniform(-7.5, 7.5, n) # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n) # Random Gaussian noise
y = f_true(X) + e

import matplotlib.pyplot as plt # For all our plotting needs
plt.figure()
# Plot the data
plt.scatter(X, y, 12, marker='o')
# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3 # Fraction of examples to sample for the test set
val_frac = 0.1 # Fraction of examples to sample for the validation set
# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

# X float(n, ): univariate data
# d int: degree of polynomial
def polynomial_transform(X, d):
    return np.vander(X, d)

# Phi float(n, d): transformed data
# y float(n, ): labels
def train_model(Phi, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T), y)

# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): linear regression model
def evaluate_model(Phi, y, w):
    for i in range(len(y)):
        inner = Phi @ w
        f = np.sum(np.power(np.subtract(y, inner), 2))
    return f / len(y)

w = {} # Dictionary to store all the trained models
validationErr = {} # Validation error of the models
testErr = {} # Test error of all the models
for d in range(3, 25, 3): # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d) # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn) # Learn model ontraining data

    Phi_val = polynomial_transform(X_val, d) # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d]) # Evaluate modelon validation data

    Phi_tst = polynomial_transform(X_tst, d) # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d]) # Evaluate model on test data
# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)
plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])


#Discussion: d = 18 is the best choice as it has the smallest validation and test error out of all the choices

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    rows = []
    for i in X:
        cols = []
        for j in range(len(B)):
            cols.append(np.exp(-1*gamma*(pow(i - B[j],2))))
        rows.append(cols)
    return rows

# Phi float(n, d): transformed data
# y float(n, ): labels
# lam float : regularization parameter
def train_ridge_model(Phi, y, lam):
    p = np.transpose(Phi)
    temp = p @ Phi
    inner = temp + (np.array(np.identity(len(y))) * lam)
    invinner = np.linalg.inv(inner)
    outer = p @ y
    som = invinner @ outer
    return som

w = {}  # Dictionary to store all the trained models
validationErr = {}  # Validation error of the models
testErr = {}  # Test error of all the models
lamda = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
rep = [i for i in range(0, len(lamda))]
for index, l in enumerate(lamda):  # Iterate over lambda values
    Phi_trn = radial_basis_transform(X_trn, X_trn, gamma=0.1)  # Transform training data
    w[index] = train_ridge_model(Phi_trn, y_trn, lam=l)  # Learn model on training data

    Phi_val = radial_basis_transform(X_val, X_trn, gamma=0.1)  # Transform validation data
    validationErr[index] = evaluate_model(Phi_val, y_val, w[index])  # Evaluate model on validation data

    Phi_tst = radial_basis_transform(X_tst, X_trn, gamma=0.1)  # Transform test data
    testErr[index] = evaluate_model(Phi_tst, y_tst, w[index])  # Evaluate model on test data
# Plot all the models
plt.figure()
plt.ylim(25, 65)
plt.plot(rep, validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(rep, testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lamda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(rep, lamda, fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
for index, l in enumerate(lamda):
    X_d = radial_basis_transform(x_true,X_trn,gamma=0.1)
    y_d = X_d @ w[index]
    plt.plot(x_true, y_d, marker='None', linewidth=2)
plt.legend(['true'] + lamda,loc='center left', bbox_to_anchor=(0.5, 1.05))
plt.show()
#The ideal value of lambda is probably close to .00001, but with the current values of lambda, it's .001
#Discussion: the linearity of the model increases with the values of lambda as the greater the values of
#lambda, the greater the validation and test error. So, a lambda value of .001 is most accurate of the current range
# of values according to the true values
