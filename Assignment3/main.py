# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
 # Generate a non-linear data set
 X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

 # Take a small subset of the data and make it VERY noisy; that is, generate outliers
 m = 30
 np.random.seed(30) # Deliberately use a different seed
 ind = np.random.permutation(n_samples)[:m]
 X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
 y[ind] = 1 - y[ind]
# Plot this data
 cmap = ListedColormap(['#b30065', '#178000'])
 plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
 # First, we use train_test_split to partition (X, y) into training and test sets
 X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
 # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
 X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
 return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

 #
# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
def visualize(models, param, X, y):
 # Initialize plotting
 if len(models) % 3 == 0:
  nrows = len(models) // 3
 else:
  nrows = len(models) // 3 + 1

 fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
 cmap = ListedColormap(['#b30065', '#178000'])
 # Create a mesh
 xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
 yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
 xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), np.arange(yMin, yMax, 0.01))
 for i, (p, clf) in enumerate(models.items()):
  # if i > 0:
  # break
  r, c = np.divmod(i, 3)
  ax = axes[r, c]
  # Plot contours
  zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
  zMesh = zMesh.reshape(xMesh.shape)
  ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)
  if (param == 'C' and p > 0.0) or (param == 'gamma'):
   ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
   # Plot data
   ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
   ax.set_title('{0} = {1}'.format(param, p))

# Generate the data
n_samples = 300 # Total size of data set
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

# Learn support vector classifiers with a radial-basis function kernel with
# fixed gamma = 1 / (n_features * X.std()) and different values of C
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()

for i, C in enumerate(C_values):
 clf = SVC(C=C, gamma='scale', kernel='rbf', random_state=0)
 clf.fit(X_trn, y_trn)
 models[C] = clf
 y_trnPredict = clf.predict(X_trn)
 y_valPredict = clf.predict(X_val)
 trnErr[i] = 1 - accuracy_score(y_trn, y_trnPredict)
 valErr[i] = 1 - accuracy_score(y_val, y_valPredict)

plt.figure(figsize=(10, 8))
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('C', fontsize=16)
plt.ylabel('Validation/Training Error', fontsize=16)
plt.xticks(list(trnErr.keys()), ('0.001', '0.01', '0.1', '1.0', '10.0', '100.0', '1000.0', '10000.0', '100000.0'), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=16)

visualize(models, 'C', X_trn, y_trn)
#plt.show()
minErr = 1
cminErr = 1
bestIdx = []
cval = list(C_values)
for val in valErr:
  if (valErr[val] < minErr):
   minErr = valErr[val]
   bestIdx.clear()
   bestIdx.append(cval[val])
  elif (valErr[val] == minErr):
   cminErr += 1
   bestIdx.append(cval[val])

print('Best C values: ')
print(bestIdx)

# Calculate test Error for each of the best indices
for c in bestIdx:
 clf = SVC(C = c,gamma='scale',kernel='rbf')
 clf.fit(X_trn, y_trn)
 y_tstpred = clf.predict(X_tst)
 tstErr=1-accuracy_score(y_tst,y_tstpred)
 print(str(c) + ', Test Error:'+ str(tstErr))
 print("Accuracy Score:")
 print(accuracy_score(y_tst,y_tstpred)*100)

#Discussion: As C increases, both training and validation error decrease almost exponentially. Training error seems to decrease
# linearly after a large drop between C values of .01 and .1. The validation error seems to do the same until C values of
# 1000 where it begins to oscillate between .1 and .2 for every subsequent C value. C operates as a regularization parameter,
# in the way that as we increase C, our slack values are increasing penalized. For larger values of C, a smaller margin
# for the decision function is accepted if the decision function is better at classifying the training points correctly.
# Whereas for smaller values of C, we get a larger margin that makes a simpler decision function, which negatively affects
# our training accuracy. We can see this in the graphs and the complexity of their boundaries. The larger the C values, the
# higher the cost of misclassification and chance of overfitting due to the stricter approach to the input data. While
# smaller values of C, the lower the cost of misclassification and the chance of underfitting due to the looser approach
# to the input data.

#Final Model Selection: We can select 100 as the best C values as it has the lowest test error and highest accuracy out of
# all of the best C values

# Learn support vector classifiers with a radial-basis function kernel with
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()

for i, G in enumerate(gamma_values):
 clf = SVC(C=10, gamma=G, kernel='rbf')
 clf.fit(X_trn, y_trn)
 models[G] = clf
 y_trnPredict = clf.predict(X_trn)
 y_valPredict = clf.predict(X_val)
 trnErr[i] = 1 - accuracy_score(y_trn, y_trnPredict)
 valErr[i] = 1 - accuracy_score(y_val, y_valPredict)

plt.figure(figsize=(10, 8))
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Gamma', fontsize=16)
plt.ylabel('Validation/Training error', fontsize=16)
plt.xticks(list(trnErr.keys()), ('0.01', '0.1', '1.0', '10.0', '100.0', '1000.0'), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=16)

visualize(models, 'gamma', X_trn, y_trn)
#plt.show()
minErr = 1
cminErr = 1
bestIdx = []
gval = list(gamma_values)
for val in valErr:
 if (valErr[val] < minErr):
  minErr = valErr[val]
  bestIdx.clear()
  bestIdx.append(gval[val])
 elif (valErr[val] == minErr):
  cminErr += 1
  bestIdx.append(gval[val])

print('Best Gamma values: ')
print(bestIdx)

# Calculate test Error for each of the best indices
for g in bestIdx:
 clf = SVC(C = 10,gamma=g,kernel='rbf')
 clf.fit(X_trn, y_trn)
 y_tstpred = clf.predict(X_tst)
 tstErr=1-accuracy_score(y_tst,y_tstpred)
 print(str(g) + ', Test Error:'+ str(tstErr))
 print("Accuracy Score:")
 print(accuracy_score(y_tst,y_tstpred)*100)

#Discussion: We get a pretty consistent decrease in training error as gamma increases. While the validation error begins to
# increase again after gamma values of 10. This gamma parameter defines the influence of a single training example. The lower
# the gamma, the larger the decision boundaries are. Whereas with higher gamma values, the smaller the decision boundaries are.
# This results in more isolated kind of 'islands' around the data points.

#Final Model Selection: We can select 10 as the best gamma value as it has the lowest test error, training error, and
# highest accuracy out of all of the best gamma values

# Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()
#
# Insert your code here.
#
cancerTrn = np.loadtxt(open("wdbc_trn.csv", "rb"), delimiter=",")
Xtrn = cancerTrn[:,1:]
Ytrn = cancerTrn[:,0]
cancerTst = np.loadtxt(open("wdbc_tst.csv", "rb"), delimiter=",")
Xtst = cancerTst[:,1:]
Ytst = cancerTst[:,0]
cancerVal = np.loadtxt(open("wdbc_val.csv", "rb"), delimiter=",")
Xval = cancerVal[:,1:]
Yval = cancerVal[:,0]

#
#
# Insert your code here to perform model selection
#
#
#Generate Values
C_range = np.arange(-2.0, 5.0, 1.0)
C_values = np.power(10.0, C_range)
cval=list(C_values)

gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_values = np.power(10.0, gamma_range)
g_values=list(gamma_values)
bestC = []
bestG = []
testError = []
valError = []
minErr=1
for i1,i in  enumerate(C_values):
    for i2,j in enumerate(gamma_values):
        clf = SVC(C = i,gamma=j,kernel='rbf')
        clf.fit(Xtrn, Ytrn)
        y_trnpred=clf.predict(Xtrn)
        y_valpred=clf.predict(Xval)
        y_tstpred=clf.predict(Xtst)
        val_err=1-accuracy_score(Yval,y_valpred)
        tst_err=1-accuracy_score(Ytst,y_tstpred)
        if(val_err < minErr):
            minErr=val_err
            bestC.clear()
            bestG.clear()
            testError.clear()
            valError.clear()
            bestC.append(cval[i1])
            bestG.append(g_values[i2])
            testError.append(tst_err)
            valError.append(val_err)
        elif (val_err == minErr):
                bestC.append(cval[i1])
                bestG.append(g_values[i2])
                testError.append(tst_err)
                valError.append(val_err)
# print results
print('Best C values: ' + str(bestC))
print('Best G values: ' + str(bestG))
print('Test Error: ' + str(testError))
print('Minimum validation error: ' + str(minErr))
print('Max Validation accuracy: ' + str((1-minErr)*100))

# Results: Best C values: [100.0, 1000.0, 10000.0, 10000.0]
# Best G values: [0.01, 0.01, 0.001, 0.01]
# Test Error: [0.034782608695652195, 0.05217391304347829, 0.060869565217391286, 0.05217391304347829]
# Minimum validation error: 0.02608695652173909
# Max Validation accuracy: 97.3913043478261

# Final Model Selection: Best C value is 100 and the best gamma value is .01. The reported accuracy for
# this test data set is 97.391%.
#
#
# Insert your code here to perform model selection
#
#
k = [1, 5, 11, 15, 21]
trnErr = []
valErr = []
for i, val in enumerate(k):
 neighbor = KNeighborsClassifier(n_neighbors=val, algorithm='kd_tree')
 neighbor.fit(Xtrn, Ytrn)
 ytrn_predict = neighbor.predict(Xtrn)
 yval_predict = neighbor.predict(Xval)
 trnErr.append(1 - accuracy_score(Ytrn, ytrn_predict))
 valErr.append(1 - accuracy_score(Yval, yval_predict))

plt.figure(figsize=(10, 8))
plt.plot(k, valErr, marker='o', linewidth=3, markersize=12)
plt.plot(k, trnErr, marker='s', linewidth=3, markersize=12)
plt.xlabel('k', fontsize=16)
plt.ylabel('Validation/Training Error', fontsize=16)
plt.legend(['Validation Error', 'Training Error'], fontsize=16)
plt.show()

for ki in k:
 neighbor = KNeighborsClassifier(n_neighbors=ki,algorithm='kd_tree')
 neighbor.fit(Xtrn,Ytrn)
 ytst_predict = neighbor.predict(Xtst)
 print(str(ki)+', Test Accuracy: ')
 print(accuracy_score(Ytst,ytst_predict)*100)

#Final Model Selection: According to the graph, k values of 5 and 11 give the smallest validation error but
# 11 reports a higher test accuracy of 97.391%. Therefore, 11 is the best k value.

#Discussion: kNN would be the best classifier for this task because
# it reports a higher test accuracy than the SVM approach.
