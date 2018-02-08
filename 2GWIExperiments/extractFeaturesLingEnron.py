"""
This code contains auxiliar functions to reproduce the 2GWI experiment over the
Enron and Ling Datasets in Section 5 of the paper
"""

import os
import random
import shutil
from shutil import copyfile
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import time
import codecs

def cleanFolders(datasetDirectory):

    # check if dataset is empty and copy original emails from DatasetLing/EnronCopy to datasetDirectory
    src = "DatasetLingCopy/" # LingSpam
    #src = "DatasetEnronCopy/" # Enron
    if os.listdir(datasetDirectory) == []:
        print("Copying emails from: ",src," to: ",datasetDirectory)
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, datasetDirectory)

    # Remove files from Test folder
    folder = "Test/"
    if os.listdir(datasetDirectory) != []:
        print("Removing files from Test...")
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    # Remove files from Train folder
    folder = "Train/"
    if os.listdir(datasetDirectory) != []:
        print("Removing files from Train...")
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

def splitTrainAndTest(q):
    datasetDirectory = "Dataset/"
    cleanFolders(datasetDirectory)
    numberOfFiles = next(os.walk(datasetDirectory))[2]
    numberOfFiles = len(numberOfFiles)
    # Set number of email to Test and Train depending division coeficient q
    numberOfFilesTestSet = round(q * float(numberOfFiles))
    numberOfFilesTrainSet = numberOfFiles - numberOfFilesTestSet

    print("Number of files:", numberOfFiles)
    print("Files for test: ", numberOfFilesTestSet)
    print("Files for train: ", numberOfFilesTrainSet)
    print("Total Train + Test: ", numberOfFilesTestSet+numberOfFilesTrainSet)

    # Copy emails to test set, selects emails randomly
    for i in range(numberOfFilesTestSet):
        randomFile = random.choice(os.listdir(datasetDirectory))
        os.rename(datasetDirectory+randomFile,"Test/"+randomFile)
    # Copy emails to train set, selects emails randomly
    for i in range(numberOfFilesTrainSet):
        randomFile = random.choice(os.listdir(datasetDirectory))
        os.rename(datasetDirectory+randomFile,"Train/"+randomFile)

def make_Dictionary(train_dir,n):
    print("Creating dictionary...")
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m: # lingspam emails
        #with open(mail,encoding="ISO-8859-1") as m: # enron emails
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    # Non-word removal
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(n)
    dictionary.append(str("SpamOrNot")) # we add this column to add the emails labels (it is not part of the dictionary)
    return dictionary

def extract_features(mail_dir,dictionary,n):
    print("Extracting features...")
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    n = n+1
    print(n)
    features_matrix = np.zeros((len(files),n),dtype=int)
    docID = 0;
    for fil in files:
      with open(fil) as fi: # ling spam emails
      #with open(fil,encoding="ISO-8859-1") as fi: # enron emails
        str1 = fi.name
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  #features_matrix[docID,wordID] = words.count(word)
                  features_matrix[docID,wordID] = 1.0
                if i==len(dictionary)-1:
                   c = i
                   spamOrNot = str1.find("sp") # Lingspam (if email name contains sp label is 1.0)
                   # spamOrNot = str1.find("spam") # Enron (if emails name contains spam label is 1.0)
                   if int(spamOrNot)>int(1):
                      features_matrix[docID,c] = 1.0
        docID = docID + 1
    return features_matrix

def getTrainAndTest(q,n):
    splitTrainAndTest(q)

    # Create a dictionary of words with its frequency
    train_dir = 'Train'
    test_dir = 'Test'
    dictionary = make_Dictionary(train_dir,n)
    print(dictionary)

    # Prepare feature vectors per training mail and its labels
    xTrain = extract_features(train_dir,dictionary,n)
    print("x_train: ",xTrain, " shape: ", xTrain.shape)
    yTrain = xTrain[:,-1]
    print("y_train: ",yTrain, " shape: ",yTrain.shape)
    xTrain =  np.delete(xTrain,-1,axis=1)

    xTest = extract_features(test_dir,dictionary,n)
    print("x_test: ",xTest, " shape: ", xTest.shape)
    yTest = xTest[:,-1]
    print("y_test: ",yTest, " shape: ", yTest.shape)
    xTest =  np.delete(xTest,-1,axis=1)

    return xTrain, xTest, yTrain, yTest
