# Experiments Section 5

Instructions on how to reproduce the first two experiments in section 5 of the paper.

## Prerequisites

Following packages need to be installed

```
numpy
pandas
sklearn
joblib
multiprocessing
scipy
seaborn
matplotlib
```
* Python 3.x.


#### Make sure that a folder data containing UciData.csv file is present on execution directory.


### Running

#### Experiment comparing ACRA and ACRA with MC and sequentiality

Configuration in AcraVsAcraMC.py:
* n: the number of words to attack
* it: number of experiments to do
* var: variance parameter
* ut: utility matrix
* m: mc size
* q: division ratio Test and Train set


```
python AcraVsAcraMC.py
```
The program write the results into a .csv file in results folder. Results can be analysed using plotsArcraVsAcraMC.R


#### Experiment comparing ACRA and parallel ACRA

Configuration in AcraVsParAcra.py:
* n: the number of words to attack
* it: number of experiments to do
* var: variance parameter
* ut: utility matrix
* m: mc size
* q: division ratio Test and Train set


```
python AcraVsParAcra.py
```
The program write the results into a .csv file in results folder. Results can be analysed using plotsArcraVsParAcra.R 
