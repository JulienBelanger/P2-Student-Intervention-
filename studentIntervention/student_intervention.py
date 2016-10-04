# Import libraries
import numpy as np
import pandas as pd
import random
import time
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV

##################################################################################################


# Read student data
student_data = pd.read_csv("student-data.csv")
print("Student data read successfully!")
# Note: The last column 'passed' is the target/label, all other are feature columns


##################################################################################################


# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
students = pd.DataFrame(student_data)
n_students = students.shape[0]
n_features = students.shape[1]  #counting the last label one
n_passed = students[students['passed']=='yes'].shape[0]
n_failed = students[students['passed']=='no'].shape[0]
grad_rate = float(n_passed)/float(n_students) * 100
print("Total number of students: {}".format(n_students))
print("Number of students who passed: {}".format(n_passed))
print("Number of students who failed: {}".format(n_failed))
print("Graduation rate of the class: {:.2f}%".format(grad_rate))
print("Number of features: {}".format(n_features))


##################################################################################################


# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print("\nFeature column(s):-\n{}".format(feature_cols))
print("Target column: {}".format(target_col))

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print("Feature values:-")
print(X_all.head())  # print the first 5 rows


##################################################################################################


# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print("Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns)))


##################################################################################################


# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
# Create Dummy variable

X_all = preprocess_features(X_all)
y_all = y_all.replace(to_replace='yes',value=1)
y_all = y_all.replace(to_replace='no',value=0)
print(y_all)
temp = list(range(n_students))
random.shuffle(temp)
X_all = X_all.reindex(temp).reset_index(drop=True)
y_all = y_all.reindex(temp).reset_index(drop=True)
X_train = X_all[:num_train]
y_train = y_all[:num_train]
X_test = X_all[num_train:]
y_test = y_all[num_train:]
print("Training set: {} samples".format(X_train.shape[0]))
print("Test set: {} samples".format(X_test.shape[0]))
# Note: If you need a validation set, extract it from within training data
##################################################################################################


# Train a model
def train_classifier(clf, X_train, y_train):
    print("Training {}...".format(clf.__class__.__name__))
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print("Done!\nTraining time (secs): {:.3f}".format(end - start))

# TODO: Choose a model, import it and instantiate an object
clf = tree.DecisionTreeClassifier()

# Fit model to training data
train_classifier(clf, X_train, y_train)  # note: using entire training set here
print(clf)  # you can inspect the learned model by printing it


##################################################################################################


# Predict on training set and compute F1 score
def predict_labels(clf, features, target):
    print("Predicting labels using {}...".format(clf.__class__.__name__))
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print("Done!\nPrediction time (secs): {:.3f}".format(end - start))
    # I have change 'yes' to nothing since I've made y_all binary
    return f1_score(target.values, y_pred)

train_f1_score = predict_labels(clf, X_train, y_train)
print("F1 score for training set: {}".format(train_f1_score))


##################################################################################################


# Predict on test data
print("F1 score for test set: {}".format(predict_labels(clf, X_test, y_test)))


##################################################################################################


# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test, size):
    print("------------------------------------------")
    print("Training set size: {}".format(len(X_train[: size])))
    train_classifier(clf, X_train[: size], y_train[: size])
    print("F1 score for training set: {}".format(predict_labels(clf, X_train[: size], y_train[: size])))
    print("F1 score for test set: {}".format(predict_labels(clf, X_test, y_test)))


##################################################################################################


# # TODO: Run the helper function above for desired subsets of training data
# train_predict(tree.DecisionTreeClassifier(),X_train,y_train,X_test,y_test, 100)
# train_predict(tree.DecisionTreeClassifier(),X_train,y_train,X_test,y_test, 200)
# train_predict(tree.DecisionTreeClassifier(),X_train,y_train,X_test,y_test, 300)
# # Note: Keep the test set constant


##################################################################################################



# TODO: Run the helper function above for desired subsets of training data
clf = svm.SVC()
train_predict(clf, X_train, y_train, X_test, y_test, 100)
train_predict(clf, X_train, y_train, X_test, y_test, 200)
train_predict(clf, X_train, y_train, X_test, y_test, 300)
# Note: Keep the test set constant


##################################################################################################



# clf = AdaBoostClassifier()
# # TODO: Run the helper function above for desired subsets of training data
# train_predict(clf, X_train, y_train, X_test, y_test, 100)
# train_predict(clf, X_train, y_train, X_test, y_test, 200)
# train_predict(clf, X_train, y_train, X_test, y_test, 300)
# # Note: Keep the test set constant


##################################################################################################



# clf = BernoulliNB()
# # TODO: Run the helper function above for desired subsets of training data
# train_predict(clf, X_train, y_train, X_test, y_test, 100)
# train_predict(clf, X_train, y_train, X_test, y_test, 200)
# train_predict(clf, X_train, y_train, X_test, y_test, 300)
# # Note: Keep the test set constant


##################################################################################################



# clf = SGDClassifier()
# # TODO: Run the helper function above for desired subsets of training data
# train_predict(clf, X_train, y_train, X_test, y_test, 100)
# train_predict(clf, X_train, y_train, X_test, y_test, 200)
# train_predict(clf, X_train, y_train, X_test, y_test, 300)
# # Note: Keep the test set constant


##################################################################################################


# TODO: Fine-tune your model and report the best F1 score
print()
print()
print()

svc = svm.SVC()
parameters = [{'C':[1e-10, 1e-5, 1e-4, 1e-3, 1**(-2.2), 1**(-2.1), 1e-2, 1**(-1.9), 1**(-1.8), 1e-1, 1], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':['auto',0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
clf = GridSearchCV(svc, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(f1_score(y_test, clf.predict(X_test)))
