# 2-GWI Experiments

Instructions on how to reproduce the 2GWI experiments in section 5 of the paper.

## UCI Dataset


### Prerequisites

Following packages need to be installed

```
numpy
pandas
sklearn
joblib
multiprocessing
scipy
```
* Python 3.x.


#### Make sure that a folder data containing UciData.csv file is present on execution directory.

#### Make sure that the file acra_tools.py is present on execution directory.


### Running
Configuration in 2GWI-UCI.py:
* n: the number of words to attack
* it: number of experiments to do
* var: variance parameter
* ut: utility matrix
* m: mc size
* q: division ratio Test and Train set


```
python 2GWI-UCI.py
```
The program write the results into a .csv file in results folder. Results can be analysed using metrics.R


## Enron and Ling Datasets

Experiments over Ling and Enron datasets are slightly different from those in
UCI database as the number of words in the dictionary of these experiments is
not fixed.

### Prerequisites

Following packages need to be installed

```
pip install numpy
pip install pandas
pip install sklearn
```
* Python 3.x.


#### 1. Download datasets
The Enron-Spam Data Set is available at https://www.cs.cmu.edu/~enron and the Ling-Spam Data Set at http://csmining.org/index.php/ling-spam-datasets.html.
#### 2. Create folders
Create the folders Dataset/, Test/ and Train/. If you want to use Ling dataset create the folder DatasetLingCopy/ and for Enron create DatasetEnronCopy/.
```
mkdir Dataset
mkdir Test
mkdir Train
mkdir DatasetLingCopy
# mkdir DatasetEnronCopy
```
#### 3. Copy the Ling/Enron dataset emails into DatasetLingCopy/DatasetEnronCopy
#### 4. Make sure to uncomment the parts in extractFeaturesLingEnron.py of each specific dataset Ling or Enron (default is for Ling).

## Running
Configuration in 2GWI-LingEnron.py:
* n: the number of words to attack
* it: number of experiments to do
* var: variance parameter
* ut: utility matrix
* m: mc size
* q: division ratio Test and Train set
* nWords: number of words for dictionary

```
python 2GWI-LingEnron.py
```
The program write the results into a .csv file in results folder. Results can be analysed using metrics.R
