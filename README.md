# ACRA Experiment

This repository contains the code and data necessary to reproduce the paper "Adversarial Classification: an Adversarial Risk Analysis approach".
In particular it contains three different folders:

* experimentSec3.4: This folder contains the code and data needed to reproduce the experiment on section 3.4 of the paper.

* experimentsSec5: This folder contains the code and data needed to reproduce the first two  experiments on section 5 of the paper.

* 2GWIExperiments: This folder contains the code and data needed to reproduce the last experiment on section 5 of the paper.

For most experiments, we used the Spambase Data Set from the UCI Machine learning repository, located in the `data` folder. This dataset consists of 4601 emails, out of which 1813 are spam (a 39.4% spam prevalence). For each email, the database contains information about 54 relevant words. We use the bag-of-words representation with binary features indicating the presence (1) or not (0) of such words, i.e. each email is assimilated with a vector of 0's and 1's of dimension 54.

For some experiments, we also used the Enron-Spam and Ling-Spam datasets. Documentation about these datasets available at https://www.cs.cmu.edu/~enron and http://csmining.org/index.php/ling-spam-datasets.html, respectively.
