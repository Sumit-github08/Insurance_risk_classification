# **Prudential Life Insurance Assessment**

***Predicting Insurance Risk Level of an insured based on their medical/personal details.***

![image](https://storage.googleapis.com/kaggle-competitions/kaggle/4699/media/iStock_insurancehands300.png)

# **Introduction**


The importance of correct risk prediction cannot be overemphasized in Life insurance context. Life insurance company have to be careful about whom to insure and whom not to in order to stay financially solvent. While there is no perfect formula to determine the insurability, Actuarial tables have been traditionally used but they are quite time consuming. Predictive Analytics offers a promising alternative to the risk prediction task. 
Prudential is a US Insurer and is into Life insurance for last 140 years. There’s pattern observed by the company that on average only 40% of US population is having insurance. This is because of the time taking process that involves classification of each individual risks according to their medical history, background. Prudential wants to make it quicker and less labor intensive for new and existing customers to get a quote while maintaining privacy boundaries.


# **Business problem**

The goal of this problem is to develop a simplified predictive model for accurately determining the Risk level of life insurance applicants to make decision on their insurance approvals. In our task, we have 8 Risk levels with 1 being the lowest and 8 highest.

# **ML formulation of the business problem**

To solve the business problem using data science, it is needed to pose that problem as classical machine learning problem. First of all, since the data has target variable, it is supervised ML problem. Further we need to predict the risk level of insured. Hence it is a multi-class classification problem. Since
 we have 8 class labels.
 
In this Supervised Machine Learning problem we will try different feature engineering hacks and use different algorithms in order to predict the individual risk level.


# **Performance Metric**

The Evaluation Metric used as a part of this Task is “Quadratic
 weighted kappa”.

** Quadratic Weighted Kappa Metric
A weighted Kappa is a metric which is used to calculate the amount of similarity between predictions and actuals. A perfect score of 1.0 is granted when both the predictions and actuals are the same.
Whereas, the least possible score is -1 which is given when the predictions are furthest away from actuals.

# **Objective/Business Constraints**

* No low-latency requirement. But the model shouldn't take more than a minute
 to predict the risk level. 
* Model interpretablility is required. Probability of predicting the Risk level
 is useful. Since model predicting the correct Risk Level has direct impact
  on insurer portfolio.
* Predicting the values as close to actual so as to get KAPPA score
 close to 1.   

# **Data Description**

In this dataset, we are provided over a hundred variables describing attributes of life insurance applicants. The dataset is self anonymous where the attributes were grouped under six heads namely product info, family info, employment info, general health measurements , medical history info, and medical keyword(yes/No). The meaning of individual attributes under these group were unknown. There are over 127 independent variables. These variables are either Discrete, continuous or categorical in nature.

•	train.csv - the training set, contains the Response values

•	test.csv - the test set, you must predict the Response variable for all rows
 in this file

•	Id : A unique identifier associated with an application.

•	Product_Info_1-7 : A set of normalized variables relating to the product
 applied for

•	Ins_Age : Normalized age of applicant

•	Ht : Normalized height of applicant

•	Wt : Normalized weight of applicant

•	BMI : Normalized BMI of applicant

•	Employment_Info_1-6 : A set of normalized variables relating to the
 employment history of the applicant.

•	InsuredInfo_1-6 : A set of normalized variables providing information about
 the applicant.

•	Insurance_History_1-9 : A set of normalized variables relating to the
 insurance history of the applicant.

•	Family_Hist_1-5 : A set of normalized variables relating to the family
 history of the applicant.

•	Medical_History_1-41 : A set of normalized variables relating to the medical
 history of the applicant.

•	Medical_Keyword_1-48 : A set of dummy variables relating to the presence of 
 absence of a medical keyword being associated with the application.

•	Response : This is the target variable, an ordinal variable relating to the
 final decision associated with an application.
