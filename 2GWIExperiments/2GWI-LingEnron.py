import pandas as pd
import numpy as np
from acra_tools import *
import time
from joblib import Parallel, delayed
import multiprocessing
from extractFeaturesLingEnron import *

def getAcraTime(email,clf,ut,var,n):
    start_time = time.time()
    ycAcra = ACRA(email,clf,ut,var,n)
    return (ycAcra, time.time() - start_time)

def getseqMCACRATime(email,clf,ut,var,n, m):
    start_time = time.time()
    ycAcraMc = seqMCACRA(email,clf,ut,var,n, m)
    return (ycAcraMc, time.time() - start_time)


def getResults(i,y_test,emailClean,email,clf,ut,var,n,m,idx):
    tmpAcra = getAcraTime(email[[i], :],clf,ut,var,n)
    tmpSeq = getseqMCACRATime(email[[i], :],clf,ut,var,n, m)
    yNbC = nbusXlabel(emailClean[[i],:], clf, ut) 
    yNb = nbusXlabel(email[[i],:], clf, ut) 
    return ([ idx, m, y_test[i], tmpAcra[0], tmpSeq[0], tmpAcra[1], tmpSeq[1], yNbC[0], yNb[0] ])
       
def getParResults(y_test,emailClean,email,clf,ut,var,n,m,idx):
    inputs = range(email.shape[0])
    numCores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=numCores)(delayed(getResults)(i,y_test,emailClean,email,clf,ut,var,n,m,idx) for i in inputs)   
    return np.array(result)

def testGWI(n,it,var,ut,m,q,nWords,lingEnron):
    for i in range(it):
        print("i: ", i)
        X_train, X_test, y_train, y_test = getTrainAndTest(q,nWords,lingEnron)
        # Train NB
        clf = trainRawNB(X_train, y_train)
        # Attack test set
        X_testAtt = sc_attack(X_test, y_test, clf, n)
        for j in m:    
            if i == 0 and j == 0.5:
                result = getParResults(y_test,X_test,X_testAtt,clf,ut,var,n,j,i)
            else:
                result = np.append(result, getParResults(y_test,X_test,X_testAtt,clf,ut,var,n,j,i), axis=0)
    # Save results into csv            
    df = pd.DataFrame(data=result,columns = [ "idx","m","y","ycACRA", "ycSeqMCACRA", "ACRATime", "SeqMCACRATime","yNbC","yNb" ])
    df.to_csv("50-2-GWIAcraVSAcraMcLingMails2.csv") 
    print("Done!")

def runExperimentGWI():
    # Configuration 
    n = 1 # number of words to attack
    it = 1 # number of experiments
    var = 0.1 # variance parameter
    ut = np.array([[1,0],[0,1]]) # utility
    m = [0.5] # mc size
    q = 0.25 # division ratio Test and Train
    nWords = 50 # number of words for dictionary
    lingEnron = 2 # 1 for Ling, 2 for Enron (Default is Ling)

    testGWI(n,it,var,ut,m,q,nWords,lingEnron)
         
if __name__ == '__main__':
    runExperimentGWI()        

