'''
Ryan Pierce
ID: 2317826
EECS 690 - Intro to Machine Learning, Python Project 1
'''

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
print ('Libraries loaded successfully!')

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print ('Dataset loaded successfully!')

# Splitting data set into two new data sets
# X will be set of inputs, y will be set of outputs


data_array = dataset.values

# For X: select all rows, but only columns indexed from 0 to 3 (inputs)
X = data_array[:, 0:4]

# For y: select all rows, but only the last column (outputs)
y = data_array[:, 4]

# Call train_test_split with X and y as arguments to get back training
# data and test data for both inputs and outputs.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 1)

# Build the model using GaussianNB and find the estimated accuracy
# Using 2-fold cross-validation (n_splits = 2)
kfold = StratifiedKFold(n_splits = 2, random_state = 1, shuffle = True)
cv_results = cross_val_score(GaussianNB(), X_train, y_train, cv = kfold, scoring = 'accuracy')

print('\nEstimated Accuracy:')
print(f'NB: {cv_results.mean()} ({cv_results.std()})\n')

# Make predictions on validation dataset
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Accuracy of prediction
print('Accuracy:')
print(accuracy_score(y_test, predictions))
print('\n')

# Confusion Matrix
print('Confusion Matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\n')

# Classification Report
print('Classification Report:\n')
print(classification_report(y_test, predictions))
print('\n')