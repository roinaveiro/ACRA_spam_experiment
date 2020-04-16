"""
This code reproduces the second experiment on section 5 of the ACRA paper.
To execute it please do: python testAcraVsAcraMC.py
"""

import pandas as pd
import numpy as np
from acra_tools import *
import time
from joblib import Parallel, delayed
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier



dataPath = "data/"
bigSpam = pd.read_csv(dataPath + "uciData.csv")
bigSpam = shuffle(bigSpam)
n = 2 # n_GWI
it = 10 # number of experiments

rf_clean = np.zeros(it)
nb_clean = np.zeros(it)
lr_clean = np.zeros(it)
nn_clean = np.zeros(it)

rf_att = np.zeros(it)
nb_att = np.zeros(it)
lr_att = np.zeros(it)
nn_att = np.zeros(it)

for i in range(it):
    print("i: ", i)
    #Split Training-Test
    X = bigSpam.drop("spam", axis=1).values
    y = bigSpam.spam.values
    q = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=q)

    '''
    # Naive Bayes
    clf = trainRawNB(X_train, y_train)
    ## Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)
    nb_clean[i] = accuracy_score(y_test, clf.predict(X_test))
    nb_att[i] = accuracy_score(y_test, clf.predict(X_testAtt))

    # Random Forest
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    ## Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)
    rf_clean[i] = accuracy_score(y_test, clf.predict(X_test))
    rf_att[i] = accuracy_score(y_test, clf.predict(X_testAtt))

    # Logistic Regression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    ## Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)
    lr_clean[i] = accuracy_score(y_test, clf.predict(X_test))
    lr_att[i] = accuracy_score(y_test, clf.predict(X_testAtt))

    # Neural Net
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    ## Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)
    lr_clean[i] = accuracy_score(y_test, clf.predict(X_test))
    lr_att[i] = accuracy_score(y_test, clf.predict(X_testAtt))
    '''

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    ## Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)
    nn_clean[i] = accuracy_score(y_test, clf.predict(X_test))
    nn_att[i] = accuracy_score(y_test, clf.predict(X_testAtt))





#df = pd.DataFrame({'nb_clean':nb_clean, 'nb_att':nb_att, 'rf_clean':rf_clean,
#    'rf_att':rf_att, 'lr_clean':lr_clean, 'lr_att':lr_att})

df = pd.DataFrame({'nn_clean':nn_clean, 'nn_att':nn_att})

df.to_csv("results/attacks_nn.csv", index=False)
