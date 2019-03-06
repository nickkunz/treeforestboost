## Preface
Included in this repo is the RMarkdown document with the code used to create this README.md (excludes this Preface section and plot image links). Also included here is the R code containing only the requirements for analytical reproduction (excludes the narrative) and a printable pdf (formated). In addition, the images folder provides all the plots generated. Enjoy!
<br><br>

## Introduction
This exercise briefly explores the topic of Marketing as it relates to Machine Learning. It broadly serves as an exploratory exercise, exclusively focusing on Classification Trees, Random Forest, and Boosting. The objective is to use these prediction methods to identify customers that have high purchase intent to direct marketing campaigns. The exercise assumes a conceptual understanding of Classification Trees, Random Forest, and Boosting. This work was completed in partial fulfillment for the course Machine Learning for Finance & Marketing at Columbia Business School in 2019 with Dr. Lentzas and is based on the paper by Moro, Cortez, and Rita, ”A Data-Driven Approach to Predict the Success of Bank Telemarketing” Decision Support Systems, Elsevier, 62:22-31, June 2014.
<br><br>

## Background
<i>”A Portuguese retail bank uses its own contact-center to do direct marketing campaigns, mainly through phone calls (tele-marketing). Each campaign is managed in an integrated fashion and the results for all calls and clients within the campaign are gathered together, in a flat file report concerning only the data used to do the phone call. ... In this context, in September of 2010, a research project was conducted to evaluate the efficiency and effectiveness of the telemarketing campaigns to sell long-term deposits. The primary goal was to achieve previously undiscovered valuable knowledge in order to redirect managers efforts to improve campaign results. ... Since this project started being analyzed in detail in September of 2010, it meant that there were available reports for about three years of telemarketing campaigns ... ”<br>

-- Moro, Cortez, and Rita, ”A Data-Driven Approach to Predict the Success of Bank Telemarketing”, 2014</i>
<br><br>

## Requirements
First, we load the libraries 'tree', 'randomForest', and 'gbm'. The 'tree' library contains the 'tree( )' function, which is required to conduct Classification Tree regression. The 'randomForest' library contains the 'randomForest( )' function, which is required to conduct Random Forest. The library 'gbm' contains the 'gbm( )' function, which is required for Boosting. <br><br>

```{r message=FALSE, warning=FALSE}
## load libraries
library(tree) # classification trees
library(randomForest)  # random forest
library(gbm)  # boosted trees
```
<br>

## Data

Here we load the data retrieved from the University of California - Irvine (UCI) Machine Learning Repository. The source tells us that the data contains customer information related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be 'yes' to subscribe or 'no' not to subscribe.

```{r}
## load data
mkt_cmpgn_data = read.table("./bank-additional/bank-additional-full.csv",
                            sep  = ";",  # separator values 
                            header = TRUE,  # create headers
                            check.names = TRUE,  # do not change header names
                            strip.white = TRUE,  # remove leading/trailing whitespace
                            stringsAsFactors = TRUE,  # strings as factor data type
                            na.strings = c("", "NA"))  # assign blank cells na
```
<br>

## Data Inspection

<h4> Feature Names </h4>

After the data has been loaded, we briefly inspect the data frame by viewing the feature names by calling the 'names( )' function.

```{r}
## view feature names
names(mkt_cmpgn_data)  
```
<br>

<h4> Dimensions </h4>

Next, we view the number of observations and features contained within the data frame by calling the 'dim( )' function. Here we see that there are 41,188 observations and 21 features, which we know the names of from the previous section.

```{r}
## view data frame dimension
dim(mkt_cmpgn_data)  
```
<br>

<h4> Preview Observations </h4>

After, we preview the first few observations to get a more nuanced understanding of what is contained within the data frame by calling the 'head( )' function.

```{r}
## preview observations
head(mkt_cmpgn_data, 5)  
```
<br>

## Data Pre-Processing

<h4> Feature Selection </h4>

After inspecting the data (and the meta data), we remove the features 'duration', 'day_of_week', 'month' and 'nr.employed'. These features are removed, as they contain information collected after contact has already been made with the client and are highly correlated to the predicted outcome of whether the client subscribed for a term deposit. Conceptually, this is a "look ahead" into the future test set and is highly inappropriate in any prediction setting.

```{r}
## remove feats
mkt_cmpgn_data$duration = NULL
mkt_cmpgn_data$day_of_week = NULL
mkt_cmpgn_data$month = NULL
mkt_cmpgn_data$nr.employed = NULL
```
<br>

<h4> Missingness </h4>

Here we replace the character string 'unknown' with NA. Subsequently, we remove those observations containing missing values by utilizing the 'na.omit( )' function. Then, we inspect the data frame dimension as an ad hoc measurement of missingness. Recall that the original data frame dimension contained 41,188 observations and 21 features. After removing the previous 4 features and observations containing NA's, the data frame now contains 30,488 observations and 17 features. 

```{r}
## replace 'unknown' char string in the data with na
mkt_cmpgn_data[mkt_cmpgn_data == "unknown" ] = NA

## remove observations containing na's
mkt_cmpgn_data = na.omit(mkt_cmpgn_data)

## view data frame dimension
dim(mkt_cmpgn_data) 
```
<br>

<h4> Transformations </h4>

<b> Binary </b>

Consider that Trees based methods are best applied by ordered categorical values. Therefore, we reduce the 'job' feature, which contains multiple job titles to contain only two values, either employed or unemployed. Similarly, we apply the same transformation to the 'marital' feature, which contains multiple marital status, to either married or single.

```{r}
## convert 'job' feat to char string data type for transformation
mkt_cmpgn_data$job = as.character(mkt_cmpgn_data$job)

## transform 'job' feat from multiple titles to two categories, either employed or unemployed
mkt_cmpgn_data$job[mkt_cmpgn_data$job != "unemployed"] = "employed"

## convert 'marital' feat to char string data type for transformation
mkt_cmpgn_data$marital = as.character(mkt_cmpgn_data$marital)

## transform 'martial' feat from marital titles to two categories, either married or single
mkt_cmpgn_data$marital[mkt_cmpgn_data$marital != "married"] = "single"

```
<br>

<b> Ordinal </b><br>

In addition, we transform the 'education' feature to a hierarchical ordered numeric dummy feature, taking 6 increasing values commensurate with the levels of education observed. Here we assign the 'illiterate' values to 0, the 'basic.4y' to 1, 'basic.6y' to 2, 'basic.9y' to 3, 'high.school' to 4, 'professional.course' to 5, and 'university.degree' to 6. As a final step in this transformation, we convert the feature 'education' to numeric.

```{r}
## convert 'education' feat to char string data type for transformation
mkt_cmpgn_data$education = as.character(mkt_cmpgn_data$education)

## preview education levels
unique(mkt_cmpgn_data$education)

## transform 'education' feat from char string to num ordinals
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "illiterate"] = 0
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "basic.4y"] = 1
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "basic.6y"] = 2
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "basic.9y"] = 3
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "high.school"] = 4
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "professional.course"] = 5
mkt_cmpgn_data$education[mkt_cmpgn_data$education == "university.degree"] = 6

## convert 'education' feat to num data type for analysis
mkt_cmpgn_data$education = as.numeric(mkt_cmpgn_data$education)
```
<br>

<h4> Data Types </h4>

Recall that the 'tree( )' function in R only takes numeric and factor data types as inputs. Therefore, we transform any and all character features in the data frame into factors. One way to achieve this is to utilize the 'as.factor( )' to manually transform the data type for each feature in the data frame. However, we know that as a default, R treats all features containing character strings as factors unless specified otherwise. We utilize this aspect of the programming language in this example to quickly convert all features containing character strings to factors by with the 'unclass( )' function and returning the result to the original data frame. After, we confirm that the data types have been converted correctly.

```{r}
## convert data types
mkt_cmpgn_data = as.data.frame(unclass(mkt_cmpgn_data))

## confirm data types
str(mkt_cmpgn_data)  
```
<br>

## Descriptive Statistics
As a measure of analytical prudence, we provide brief descriptive statistics for the features contained within the data frame 'mkt_cmpgn_data'.

```{r}
## descriptive stats
summary(mkt_cmpgn_data)
```
<br>

## Data Splitting

Now that we have clean data, we split it into training and test sets. This allows us to compare and assess our results from Classification Trees, Random Forest, and Boosting. A prior step before we move forward with splitting our data into a training and test set, is specifying a random number with the function 'set.seed( )'. This allows our results to be reproducible for further analysis. After we begin at an established random number, we conduct splitting by calling the function 'sample( )'.

```{r}
## random number
set.seed(1)  

## training and test split
train = sample(1:nrow(mkt_cmpgn_data), nrow(mkt_cmpgn_data) / 2)  # create training set
test = mkt_cmpgn_data[-train, ]  # create test set 
y.test = mkt_cmpgn_data$y[-train]  # create test y 

```
<br>

## Classification Trees

<h4> Training </h4>

Here we begin to train the Classification Tree utilizing the 'tree( )' function. The function is simple, where the first argument contains the formula used in prediction (the dependent Y variable is the 'y' feature in data frame 'mkt_cmpg_data', the independent X variables are all other features, called by the period '.' before the tilde), the second argument specifies the data frame to call, and the third argument subsets 'mkt-cmpgn_data' to limit training only to the training set.

```{r}
## tree training
tree = tree(y ~.,  # training formula
            data = mkt_cmpgn_data,  # data frame
            subset = train)  # limit training to training set
```
<br>

<h4> Feature Importance </h4>

To help better interpret the Classification Tree results, we plot and inspect the tree diagram. Interpreting the results, the most important feature in this classification is 'euribor3m', where the observations are split at 1.2395 (first split). The second most important feature is 'pdays', where the observations are split at 513 (second split). Also, we see the 'euribor3m' feature again (second split). However, the split occurs at a different level in the observations at 3.1675. Recall the goal of this exercise is to predict whether or not the client would subscribe for a bank term deposit. Observe the outcomes, 'yes' or 'no'. Notice that the results 'yes' (the result we are interested in) fall under the tree at less than 1.2395 in 'euribor3m' and of those, only on those which are also less than 513 in 'pdays' result in 'yes'. Meaning, the lower the average inter-bank interest rate at which European banks are prepared to lend to one another ('euribor3m') and the lower the number of days that passed by after the client was last contacted from a previous campaign ('pday') were the most influential indicators of marketing effectiveness according to the Classification Tree model. This is important to consider when fitting a the Classification Tree model and at the next step in prediction.

```{r}
## tree plotting
plot(tree)
text(tree, cex = 0.75)
```
![tree.png](https://github.com/nickkunz/treeforestboost/blob/master/images/tree.png)

<br>

<h4> Prediction </h4>

After we trained the model, we utilize it for making predictions on the test set with the 'predict( )' function. Note that the 'type' argument is utilized in order to specify the prediction type. In this case, the type of prediction we're interested in is classification, indicated by the term "class".

```{r}
## tree prediction
tree_predict = predict(tree, 
                       newdata = test,  # test set
                       type = "class")  # classification
```
<br>

<h4> Performance Evaluation </h4>

Next, we evaluate the model's performance by calling the 'table( )' function and calculate the Test MSE. The results indicate a Test MSE of roughly 11.3%. Meaning, the model achieves a roughly 88.7% accuracy on whether or not the client would subscribe for a bank term deposit. Cool!

```{r}
## tree prediction results
table(tree_predict, y.test)

## calculate test mse
1 - (13180 + 339) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy
(13180 + 339) / (nrow(mkt_cmpgn_data) / 2)
```
<br>

<h4> Alternatives </h4>

<b> Gini </b><br>

As an alternative tree splitting method, the following serves as an experiment to test if we can improve the model's predictive accuracy by simply specifying a different splitting criteria. The following succinctly demonstrates the repetition of all previous analysis. However, uses 'gini' as the splitting criteria. The results indicate a Test MSE of roughly 12.4%. Meaning, the model achieves a roughly 87.6% accuracy on whether or not the client would subscribe to a bank term deposit. In other words, a reduction in the predictive performance of the model.

```{r}
## tree training - gini
tree_gini = tree(y ~.,  # training formula
                 data = mkt_cmpgn_data,  # data frame
                 subset = train,  # limit training to training set
                 split = "gini")  # splitting criteria

## tree plotting - gini
plot(tree_gini)
text(tree_gini, cex = 0.75)

![tree_gini.png](https://github.com/nickkunz/treeforestboost/blob/master/images/tree_gini.png)

## tree prediction - gini
tree_predict_gini = predict(tree_gini, test, type = "class")

## tree prediction results - gini
table(tree_predict_gini, y.test)

## calculate test mse - gini
1 - (12764 + 591) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy - gini
(12764 + 591) / (nrow(mkt_cmpgn_data) / 2) 
```
<br>

## Random Forest

<h4> Training </h4>

Here we begin to train Random Forest utilizing the 'randomForest( )' function. The function is simple, where the first argument contains the formula used in prediction (the dependent Y variable is the 'y' feature in data frame 'mkt_cmpg_data', the independent X variables are all other features, called by the period '.' before the tilde), the second argument specifies the data frame to call, and the third argument subsets 'mkt-cmpgn_data' to limit training only to the training set. The fourth argument specifies the tuning parameter 'm' where it is set at a generally accepted square root of 'p' or the number of features in the data frame 'mkt_cmpgn_data'. Lastly, in the 'importance' argument, we specify the output of the feature importance.

```{r}
## random forest training
r_forest = randomForest(y ~.,  # training formula
                        data = mkt_cmpgn_data,  # data frame
                        subset = train,  # limit training to training set
                        mtry = sqrt(ncol(mkt_cmpgn_data)),  # tuning param 'm'
                        importance = TRUE)  # output feature importance
```
<br>

<h4> Feature Importance </h4>

To help better interpret the Random Forest training results, we create a table and a corresponding plot that displays each features influence on the model. Interpreting the results, the most important feature is 'euribor3m', similar to the Classification Tree model. However the second most important feature is 'cons.price.idx', which tells us something different than the Classification Tree model, where the second most important feature was 'pdays' the number of days that passed by after the client was last contacted from a previous campaign. Meaning, the average inter-bank interest rate at which European banks are prepared to lend to one another ('euribor3m') and consumer price index ('cons.price.idx') were the most influential indicators of marketing effectiveness according to the Random Forest model. The following plot is measured by the decrease in mean accuracy.

```{r}
## plot random forest feature importance 
varImpPlot(r_forest, 
           type = 1,
           sort = TRUE, 
           n.var = nrow(r_forest$importance))

## preview random forest feature importance 
importance(r_forest)
```
![forest_var_imp.png](https://github.com/nickkunz/treeforestboost/blob/master/images/forest_var_imp.png)

<br>

<h4> Prediction </h4>

After we trained the model, we utilize it for making predictions on the test set with the 'predict( )' function. Note that the 'type' argument is utilized in order to specify the prediction type. Similarly to the previous examples, the type of prediction we're interested in is classification, indicated by the term "class".

```{r}
## random forest prediction
r_forest_predict = predict(r_forest, 
                           newdata = test,  # test set
                           type = "class")  # classification

```
<br>

<h4> Performance Evaluation </h4>

Next, we evaluate the model's performance by calling the 'table( )' function and calculate the Test MSE. Our results indicate a Test MSE of roughly 11.6%. Meaning, the model achieves a roughly 88.4% accuracy on whether or not the client would subscribe for a bank term deposit. There was no improvement, but rather a marginal reduction in the predictive performance in this model when compared to Random Forest.

```{r}
## random forest prediction results 
table(r_forest_predict, y.test)

## calculate test mse
1 - (12898 + 580) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy 
(12898 + 580) / (nrow(mkt_cmpgn_data) / 2) 
```
<br>

## Boosting

<h4> Pre-Processing </h4>

Before we proceed with Boosting with the 'gbm( )' function, the algorithm requires that our Y prediction 'y' is numeric. Since our the classification results we are predicting is binary, either 'yes' or 'no', we transform those observations in the 'y' feature to either 1 or 0, respectively. We apply this transformation to all data sets.

```{r}
## training set

## convert 'y' feat to char string data type for transformation
mkt_cmpgn_data$y = as.character(mkt_cmpgn_data$y)

## transform 'y' feat from char string (yes/no) to binary (1/0)
mkt_cmpgn_data$y[mkt_cmpgn_data$y == "yes"] = 1
mkt_cmpgn_data$y[mkt_cmpgn_data$y == "no"] = 0

## convert 'y' feat to num
mkt_cmpgn_data$y = as.numeric(mkt_cmpgn_data$y)

## test set

## convert 'y' feat to char string data type for transformation
test$y = as.character(test$y)

## transform 'y' feat from char string (yes/no) to binary (1/0)
test$y[test$y == "yes"] = 1
test$y[test$y == "no"] = 0

## convert 'y' feat to num
test$y = as.numeric(test$y)

## test y

## convert 'y' feat to char string data type for transformation
y.test = as.character(y.test)

## transform 'y' feat from char string (yes/no) to binary (1/0)
y.test[y.test == "yes"] = 1
y.test[y.test == "no"] = 0

## convert 'y' feat to num
y.test = as.numeric(y.test)
```
<br>

<h4>Training</h4>

Here we begin training the Boosting model utilizing the 'gbm( )' function. The first argument contains the formula used in prediction (the dependent Y variable is the 'y' feature in data frame 'mkt_cmpg_data', the independent X variables are all other features, called by the period '.' before the tilde), the second argument specifies the data frame to call. The 'distribution' argument specified here as 'bernoulli', which is required for classification. If we were interested in regression, the argument would be more appropriately specified as 'gaussian', as a normal distribution. The argument 'n.trees' specifies the number of trees to create. In this case we somewhat arbitrarily choose 10,000. We do not test the optimal number of trees here, as it is outside of the scope of this study. However, it is a prudent number of trees and falls within the recommended range according the algorithms documentation. The argument 'interaction.depth' indicates the tree depth we establish at 3, which also falls within the recommended range. In our finally argument, we establish the tuning parameter lambda to slightly shrink our coefficient estimates, introducing bias, but in anticipation for decreased variance in the Test MSE.

```{r}
## boosting training
boost = gbm(y ~., 
            data = mkt_cmpgn_data[train, ], 
            distribution = "bernoulli",  # bernoulli distribution for classification
            n.trees = 10000,  # num of trees 
            interaction.depth = 3,  # tree depth 
            shrinkage = 0.001)  # tuning param lambda
```
<br>

<h4> Feature Importance </h4>

To help better interpret the Boosting training results, we plot each features influence on the model from its corresponding table. Exhibited here are the relative influences of each feature. Interpreting the results, the most important feature is 'euribor3m', similar to the Classification Tree and Random Forest models. However, the second most important feature is 'pday', which tells us something different than the Random Forest model, where the second most important feature was the consumer price index ('cons.price.idx'). Yet, it tells us something similar to the Classification Tree model, where the second most important feature was also 'pday'. Meaning, the average inter-bank interest rate at which European banks are prepared to lend to one another ('euribor3m') and the number of days that passed by after the client was last contacted from a previous campaign ('pday') were the most influential indicators of marketing effectiveness according to the Boosting model. Perhaps there is something to be gleaned from the similarity in the two most important features from 2 out of the 3 models? The following plot is measured by the decrease in mean accuracy.

```{r}
## preview boosting feature importance 
summary(boost, 
        method = relative.influence,
        las = 2)
```
![boost_var_imp.png](https://github.com/nickkunz/treeforestboost/blob/master/images/boost_var_imp.png)
<br>

<h4> Prediction </h4>

After we trained the model, we utilize it for making predictions on the test set. Note that the 'type' argument is utilized in order to specify the prediction type. In this case, the type of prediction we're interested in is classification. However, unlike Classification Trees and Random Forest, where our prediction was a factor indicated by the term "class", we need to specify "response" as our prediction in numerical terms. Also, because of the application of Boosting, we specify the same number of trees utilized in fitting the model (10,000). As a final step in Boosting prediction, we round our results to either 1 or 0.

```{r}
## boosting prediction
boost_predict = predict(boost,
                        newdata = test,
                        type = "response",
                        n.trees = 10000)

## round boosting predictions to binary (1/0)
boost_predict = round(boost_predict)
```
<br>

<h4> Performance Evaluation </h4>

Next, we evaluate the model's performance by calling the 'table( )' function and calculate the Test MSE. The results indicate a Test MSE of roughly 11.2%. Meaning, the model achieves a roughly 88.8% accuracy on whether or not the client would subscribe for a bank term deposit. There was a marginal improvement in the predictive performance in this model when compared to both Classification Trees and Random Forest.

```{r}
## boosting prediction results 
table(boost_predict, y.test)

## calculate test mse
1 - (13083 + 456) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy 
(13083 + 456) / (nrow(mkt_cmpgn_data) / 2) 

```
<br>

## Results & Discussion

Examining the Test MSE results from Classification Trees, Random Forest, and Boosting, the model that achieved the lowest Test MSE (highest predictive accuracy) is the Boosting model. The Boosting model resulted in a Test MSE of roughly 11.2%. The Classification Tree model performed only slightly worse with the Test MSE of roughly 11.3%. The Random Forest model had similar performance, yet slightly worse than the rest with a Test MSE of roughly 11.6%. It is important to consider that these differences are marginal and the computational cost in achieving them should also be considered. Although the Boosting model achieved the highest predictive performance, it is generally more computationally expensive than the Classification Tree model. Calculating the computational cost is beyond the scope of this discussion. However, the speed at which these results cannot be dismissed; especially in another application, marketing or otherwise. As a recommendation for this application, Classification Trees might be appropriate. With only a marginal loss in predictive accuracy, Classification Trees achieve relative predictive quality and are computationally inexpensive when compared to Random Forest and Boosting.

```{r}
## calculate test mse - trees
1 - (13180 + 339) / (nrow(mkt_cmpgn_data) / 2)

## calculate test mse - random forest
1 - (12898 + 580) / (nrow(mkt_cmpgn_data) / 2)

## calculate test mse - boosting
1 - (13083 + 456) / (nrow(mkt_cmpgn_data) / 2)
```
<br>

## Conclusion

This exercise briefly explored the topic of Marketing as it relates to Machine Learning. The study tested the predictive accuracy of whether or not a client would subscribe for a bank term deposit for a Portuguese retail bank using direct telemarketing campaigns. It focused exclusively on Classification Trees, Random Forest, and Boosting. This study serves as an exploratory exercise with commonly applied machine learning methods. More information in this regard can be found in the text "An Introduction to Statistical Learning with Applications in R" by James, Witten, Hastie, and Tibshirani (2016). With regard to the content of the study, more information can be found in the text ”A Data-Driven Approach to Predict the Success of Bank Telemarketing" by Moro, Cortez, and Rita. <br><br>
