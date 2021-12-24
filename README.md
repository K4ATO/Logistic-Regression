# Logistic-Regression
### steps:
#### 1. Select features which lead us to our `target`.
* For predicting `target` in our dataset, we need to consider 4 features from our dataset to deal with.
* Our features are `trestbps, chol, thalach, oldpeak`.
#### 2. Prepare the Data.
* Select features and target from the dataset file.
* Normalize the data.
* Separate the data to `training data` and `testing data`.
* Add Ones column for Theta zero.
* Convert the training and testing data to matrices.
* Initialize theta matrix.

#### 3. Implement the hypothesis function
* Logistic Regression hypothesis function is `Sigmoid Function`
![](https://i.imgur.com/0STr3qR.png)
#### 4. Implement the cost function
![](https://i.imgur.com/Jag6T6i.png)
#### 5. Implement the gradient descent
![](https://i.imgur.com/3frogzZ.png)
#### 6. Predict `target` values . 
* with X_test data and theta using the hypothesis function we can predict our `target` and compare it with our Y_test.

