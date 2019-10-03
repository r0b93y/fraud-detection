# Machine Learning Engineer Nanodegree
## Capstone Project: Supervised Learning for Fraud Detection
## Robert Heyse September 28, 2019


# Project Overview
Fraud detection in online payment transactions is a key problem for payment service providers. The ambition of payment service providers when implementing a fraud detection system is to identify and flag all fraudulent transactions, while keeping the false positive rate low. By flagging all fraudulent transactions, businesses will minimize their fraud loss and increase their revenue. A high false positive rate would have a negative effect on customer experience, as transactions might be blocked or an additional effort to verify a transaction might be required.1
To solve the problem at hand, I will apply two supervised learning algorithms, XGBoost and Light GBM, on the dataset. The labeled data in the training set will serve to train the algorithms. The input of the algorithm will be a combination of the categorical and numerical features in the dataset. The output of the algorithm will be the probability that a given online transaction is fraudulent.
I will use the AUROC (area under ROC curve) evaluation metric to measure the performance of different algorithms against each other.2
Data
The dataset for this project is obtained from the IEEE-CIS Fraud Detection Competition on Kaggle, and can be downloaded on the official competition page: https://www.kaggle.com/c/ieee-fraud-detection/data. The data for the competition has been provided by Vesta, one of the leading e-commerce payment solution providers, guaranteeing more than $18B in transactions annually. It comes from real-world e-commerce transactions and contains a wide range of features from device type to product features.

1 ​https://www.kaggle.com/c/ieee-fraud-detection/overview/description
2 ​https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
   
## Technologies
This project has been created in Kaggle kernels, which can be found as IPython notebooks in the project files.
- Benchmark model: Benchmark_random_forest.ipynb
- Exploratory data analysis: EDA_Fraud_v1.ipynb
- First XGBoost model: XGBoost_Fraud_v0.ipynb
- First LGB model: Fraud_LGBM_v0.ipynb
- Second LGB model: refined_Fraud_LGBM_v1.ipynb
- Final LGB model: FINAL_Fraud_LGB_Feature_eng.ipynb

The following Python libraries have been used:
- XGBoost: ​https://xgboost.readthedocs.io/en/latest/
- Light GBM: ​https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
- Seaborn: ​http://seaborn.pydata.org/
- Matplotlib: ​https://matplotlib.org/
- Numpy: ​https://numpy.org/
- Pandas: ​https://pandas.pydata.org/
- Scikit-learn: ​https://scikit-learn.org/stable/
       
