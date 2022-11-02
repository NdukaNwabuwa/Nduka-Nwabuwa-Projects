INTRODUCTION

Blossom Bank also known as BB PLC is a multinational financial services group that offers retail and investment banking, pension management, asset management and payment services, headquartered in London, UK.

PROBLEM STATEMENT

Given the present day risk of possible fraud in online transaction processing and the need to have digital infrastrutures in place for 24/7 monitoring of these online transactions; Blossom Bank wants to build a Machine Learning Model to predict online fraud to prevent loss of revenue and also save the banks image in the financial industry.

STEPS TAKEN

The dataset was provided in .csv file format and Exloratory Data Analysis (Univariate, Bivariate and Multivariate) was performed to have a better understanding of the dateset and also for better visualization. 3 ML Algorithms (Random Forest, K-Neighbors and Logistic Regression) were used to find the accuracy score. The dataset was further separated into train and test data at 80:20 ratio respectively. It was further evaluated using confusion matrix which gave the outcome of Random Forest (0 FP and 0 FN with an accuracy score-100%) as a better ML Algorithm to be deployed by Blossom Bank to predict online payment fraud.

CHALLENGES

The first method used to get the accuracy scores for the 3 ML Algorithms (Random Forest, K-Neighbors and Logistic Regression) failed as it ended up displaying values for only "Linear Regression" leaving out the other 2 ML Algorithms. A different method was used to get the accuracy scores for the 3 ML Algorithms at different intervals.
