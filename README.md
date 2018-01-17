# ACRA Experiment.

This repository contains the code and data used in the testing experiment of the ACRA algorithm. In particular it contains two jupyter notebooks:

* acraBigExperiment.ipynb: where the experiment is explained and performed.
* Analysis.ipynb: where the results are analyzed.

We use the Spambase Data Set from the UCI Machine learning repository, located in the `data` folder. This dataset consists of 4601 emails, out of which 1813 are spam (a 39.4% spam prevalence). For each email, the database contains information about 54 relevant words. We use the bag-of-words representation with binary features indicating the presence (1) or not (0) of such words, i.e. each email is assimilated with a vector of 0's and 1's of dimension 54.

We first divide the dataset into training and test sets, respectively taking 75% and 25% of the set for training and testing purposes. Each of the results presented is an average over 100 experiments performed under different training-test splits.

We trained a naive Bayes classifier using the training data, that remains unaltered by assumption. To test the ACRA algorithm against NB on tampered data, we need to simulate attacks over the test set. For this purpose, we solved the attacker's problem for each email in the test set removing the uncertainty that is not present from the attacker's point of view. We restricted ourselves to one good word insertion (1-GWI) class of attacks: the attacker can add, at most, one word.
