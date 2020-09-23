This project explores supervised classification using two different classifiers, namely the naive-Bayes classifier and the k-nearest neighbour classifier, conducted on three different datasets. The aim is to evaluate the best performance for each of the classifiers regarding each dataset. by properly tuning the parameters of each classifier so that the least error is recorded during the classification.

Here we consider the Breast Cancer (Wisconsin), Ecoli and Spambase data sets. These datasets can be downloaded from the UCI repository using the following links:
  Breast Cancer (Wisconsin)1: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
  Spambase: https://archive.ics.uci.edu/ml/datasets/spambase
  Wine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/

From the given datasets, we create a data feature matrix or a data frame. We then divide each dataset into training set (formed randomly by choosing 80% data from each classes of a dataset) and the test set (formed by the remaining 20% data of the class).

We train our model for a particular classifier using the training sets and obtain a optimum set of parameters in order to achieve the best performance. We then use our model to categorize the data points in the test set of each dataset and evaluate the performances for each dataset. Finally, we analyse and compare our results with the actual class labels of the test samples and come up with appropriate conclusions based on our comparisons.
