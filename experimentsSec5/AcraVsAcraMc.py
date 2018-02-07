"""
This code reproduces the first experiment on section 5 of the ACRA paper.
To execute it please do: python testAcraVsAcraMC.py
"""

import pandas as pd
import numpy as np
from acra_tools import *
import time
from joblib import Parallel, delayed
import multiprocessing

def getAcraTime(email,clf,ut,var,n):
    start_time = time.time()
    ycAcra = ACRA(email,clf,ut,var,n)
    return (ycAcra, time.time() - start_time)

def getseqMCACRATime(email,clf,ut,var,n, m):
    start_time = time.time()
    ycAcraMc = seqMCACRA(email,clf,ut,var,n, m)
    return (ycAcraMc, time.time() - start_time)


def getResults(i,y_test,email,clf,ut,var,n,m,idx):
    tmpAcra = getAcraTime(email[[i], :],clf,ut,var,n)
    tmpSeq = getseqMCACRATime(email[[i], :],clf,ut,var,n, m)
    return ([ idx, m, y_test[i], tmpAcra[0], tmpSeq[0], tmpAcra[1], tmpSeq[1] ])
       
def getParResults(y_test,email,clf,ut,var,n,m,idx):
    inputs = range(email.shape[0])
    numCores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=numCores)(delayed(getResults)(i,y_test,email,clf,ut,var,n,m,idx) for i in inputs)   
    return np.array(result)

    
dataPath = "data/"
bigSpam = pd.read_csv(dataPath + "uciData.csv")
bigSpam = shuffle(bigSpam)
n = 1 # n_GWI
it = 100 # number of experiments
var = 0.1 # variance parameter
ut = np.array([[1,0],[0,1]]) # utility
m = [0.25, 0.5, 0.75] # mc size

for i in range(it):
    print("i: ", i)
    #Split Training-Test
    X = bigSpam.drop("spam", axis=1).values
    y = bigSpam.spam.values
    q = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=q)

    #Train NB
    clf = trainRawNB(X_train, y_train)

    ## Attack test set
    X_testAtt = sc_attack(X_test, y_test, clf, n)
    
    for j in m:    
        if i == 0 and j == 0.25:
            result = getParResults(y_test,X_testAtt,clf,ut,var,n,j,i)
        else:
            result = np.append(result, getParResults(y_test,X_testAtt,clf,ut,var,n,j,i), axis=0)
            
df = pd.DataFrame(data=result,
                  columns = [ "idx","m","y","ycACRA", "ycSeqMCACRA", "ACRATime", "SeqMCACRATime" ])
df.to_csv("results/AcraVSAcraMc.csv")
print("Done!")
        
        

