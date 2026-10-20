---------------------------
Types of ML Systems
---------------------------
**Supervised Learning:** When human gives the correct results to ML. This like a student learning from old exams answers for a new exam.\
-- Regression and Classification is two of the most common use cases\
**Unsupervised  Learning:**  Identify meaningful patterns in a dataset.\
**Reinforcement Learning:** Make predictions by getting rewards or penalties based on actions.\
**Gen AI :** Creates content from user input.

-------- Linear Regression ----------

It finds the relationship between features and a label.
With 3 feature learning the formula should be as below

y(prediction) = b(bias) + w1a1 (weight of feature x feature val.) + w2a2 + w3a3 

bias is a parameter of the model and is calculated during training
w1 weight is a parameter of the model is calculated during training

  -- What is Loss?

  Loss measures distance between model's predictions and the actual labels.There are 5 main type of loss calculation in linear regression\
  L1: Sum (actual val. - predicted val.)\
  Mean absolute err. (MAE) : the avg of L1 ,  DIVIDE(L1,n) n->the set of exmp. The model is closer to outliers but further away from most of the other data points.\
  L2: Sum ( (actual val. - predicted val.)² )  
  Mean squared err. (MSE) : the avg of L2 , DIVIDE(L2,n) n->the set of exmp. The model is further away from outliers but closer to most of other data points  
  Root mean squared err. (RMSE): √MSE
  
-------- Gradient Descent ----------
I will continue from below page

https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent
