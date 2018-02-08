"""
This code reproduces the 2GWI experiment over the UCI Dataset in Section 5 of
the paper
"""

import pandas as pd
import numpy as np
from acra_tools import *
import time
from joblib import Parallel, delayed
import multiprocessing
from extractFeaturesLingEnron import *


def getseqMCACRA(email,clf,ut,var,n, m):
    ycAcraMc = seqMCACRA(email,clf,ut,var,n, m)
    return (ycAcraMc)


def getResults(i,y_test,emailClean,email,clf,ut,var,n,m,idx):
    tmpSeq = getseqMCACRA(email[[i], :],clf,ut,var,n, m)
    yNbC = nbusXlabel(emailClean[[i],:], clf, ut)
    yNb = nbusXlabel(email[[i],:], clf, ut)
    return ([ idx, m, y_test[i], tmpSeq, yNbC[0], yNb[0] ])

def getParResults(y_test,emailClean,email,clf,ut,var,n,m,idx):
    inputs = range(email.shape[0])
    numCores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=numCores)(delayed(getResults)(i,y_test,emailClean,\
    email,clf,ut,var,n,m,idx) for i in inputs)
    return np.array(result)


n = 2 # number of words to attack
it = 100 # number of experiments
var = 0.1 # variance parameter
ut = np.array([[1,0],[0,1]]) # utility
m = [0.5] # mc size

for i in range(it):
    print("i: ", i)
    q = 0.25 # division ratio Test and Train
    nWords = 50 # number of words for dictionary
    X_train, X_test, y_train, y_test = getTrainAndTest(q,nWords)

    # Train NB
    clf = trainRawNB(X_train, y_train)
    # Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)

    for j in m:
        if i == 0 and j == 0.5:
            result = getParResults(y_test,X_test,X_testAtt,clf,ut,var,n,j,i)
        else:
            result = np.append(result, getParResults(y_test,X_test,X_testAtt,\
            clf,ut,var,n,j,i), axis=0)

df = pd.DataFrame(data=result,\
columns = [ "idx","m","y", "ycSeqMCACRA","yNbC","yNb" ])
df.to_csv("results/2GWI-Ling") # store results to csv
print("Done!")
