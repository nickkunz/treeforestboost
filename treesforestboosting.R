## title: trees, random forest, boosting
## by: nick kunz
## date: 3/5/19

## set working directory
setwd("FILEPATH PLACEHOLDER")

## load libraries
library(tree) # classification trees
library(randomForest)  # random forest
library(gbm)  # boosted trees

## load data
mkt_cmpgn_data = read.table("./bank-additional/bank-additional-full.csv",
                            sep  = ";",  # separator values 
                            header = TRUE,  # create headers
                            check.names = TRUE,  # do not change header names
                            strip.white = TRUE,  # remove leading/trailing whitespace
                            stringsAsFactors = TRUE,  # strings as factor data type
                            na.strings = c("", "NA"))  # assign blank cells na

## view feature names
names(mkt_cmpgn_data)  

## view data frame dimension
dim(mkt_cmpgn_data)  

## preview observations
head(mkt_cmpgn_data, 5)  

## remove feats
mkt_cmpgn_data$duration = NULL
mkt_cmpgn_data$day_of_week = NULL
mkt_cmpgn_data$month = NULL
mkt_cmpgn_data$nr.employed = NULL

## replace 'unknown' char string in the data with na
mkt_cmpgn_data[mkt_cmpgn_data == "unknown" ] = NA

## remove observations containing na's
mkt_cmpgn_data = na.omit(mkt_cmpgn_data)

## view data frame dimension
dim(mkt_cmpgn_data) 

## convert 'job' feat to char string data type for transformation
mkt_cmpgn_data$job = as.character(mkt_cmpgn_data$job)

## transform 'job' feat from multiple titles to two categories, either employed or unemployed
mkt_cmpgn_data$job[mkt_cmpgn_data$job != "unemployed"] = "employed"

## convert 'marital' feat to char string data type for transformation
mkt_cmpgn_data$marital = as.character(mkt_cmpgn_data$marital)

## transform 'martial' feat from marital titles to two categories, either married or single
mkt_cmpgn_data$marital[mkt_cmpgn_data$marital != "married"] = "single"

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

## convert data types
mkt_cmpgn_data = as.data.frame(unclass(mkt_cmpgn_data))

## confirm data types
str(mkt_cmpgn_data)  

## descriptive stats
summary(mkt_cmpgn_data)

## random number
set.seed(1)  

## training and test split
train = sample(1:nrow(mkt_cmpgn_data), nrow(mkt_cmpgn_data) / 2)  # create training set
test = mkt_cmpgn_data[-train, ]  # create test set 
y.test = mkt_cmpgn_data$y[-train]  # create test y 

## tree training
tree = tree(y ~.,  # training formula
            data = mkt_cmpgn_data,  # data frame
            subset = train)  # limit training to training set

## tree plotting
plot(tree)
text(tree, cex = 0.75)

## tree prediction
tree_predict = predict(tree, 
               newdata = test,  # test set
               type = "class")  # classification

## calculate test mse
1 - (13180 + 339) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy
(13180 + 339) / (nrow(mkt_cmpgn_data) / 2)

## tree training - gini
tree_gini = tree(y ~.,  # training formula
                 data = mkt_cmpgn_data,  # data frame
                 subset = train,  # limit training to training set
                 split = "gini")  # splitting criteria

## tree plotting - gini
plot(tree_gini)
text(tree_gini, cex = 0.75)

## tree prediction - gini
tree_predict_gini = predict(tree_gini, test, type = "class")

## tree prediction results - gini
table(tree_predict_gini, y.test)

## calculate test mse - gini
1 - (12764 + 591) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy - gini
(12764 + 591) / (nrow(mkt_cmpgn_data) / 2) 

## random forest training
r_forest = randomForest(y ~.,  # training formula
                        data = mkt_cmpgn_data,  # data frame
                        subset = train,  # limit training to training set
                        mtry = sqrt(ncol(mkt_cmpgn_data)),  # tuning param 'm'
                        importance = TRUE)  # output feature importance

## plot random forest feature importance 
varImpPlot(r_forest, 
           type = 1,
           sort = TRUE, 
           n.var = nrow(r_forest$importance))

## preview random forest feature importance 
importance(r_forest)

## random forest prediction
r_forest_predict = predict(r_forest, 
                   newdata = test,  # test set
                   type = "class")  # classification

## random forest prediction results 
table(r_forest_predict, y.test)

## calculate test mse
1 - (12898 + 580) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy 
(12898 + 580) / (nrow(mkt_cmpgn_data) / 2) 

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

## boosting training
boost = gbm(y ~., 
            data = mkt_cmpgn_data[train, ], 
            distribution = "bernoulli",  # bernoulli distribution for classification
            n.trees = 10000,  # num of trees 
            interaction.depth = 3,  # tree depth 
            shrinkage = 0.001)  # tuning param lambda

## preview boosting feature importance 
summary(boost, 
        method = relative.influence,
        las = 2)

## boosting prediction
boost_predict = predict(boost,
                newdata = test,
                type = "response",
                n.trees = 10000)

## round boosting predictions to binary (1/0)
boost_predict = round(boost_predict)

## boosting prediction results 
table(boost_predict, y.test)

## calculate test mse
1 - (13083 + 456) / (nrow(mkt_cmpgn_data) / 2)

## calculate accuracy 
(13083 + 456) / (nrow(mkt_cmpgn_data) / 2) 

## final results

## calculate test mse - trees
1 - (13180 + 339) / (nrow(mkt_cmpgn_data) / 2)

## calculate test mse - random forest
1 - (12898 + 580) / (nrow(mkt_cmpgn_data) / 2)

## calculate test mse - boosting
1 - (13083 + 456) / (nrow(mkt_cmpgn_data) / 2)