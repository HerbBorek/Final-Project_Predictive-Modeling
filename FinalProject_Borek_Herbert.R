#############################################
#                                           #
# Author:     Herbert Borek                 #
# Date:       08/25/19                      #
# Subject:    Final Project                 #
# Class:      BDAT 640                      #
# Section:    01W                           #         
# Instructor: Chris Shannon                 #
# File Name:  FinalProject_Borek_Herbert.R  #
#                                           # 
#############################################


########################
# 1.  Data Preparation #
########################

#     a.  Load the dataset insurance.csv into memory.
insurance <- read.csv("insurance.csv")
#     b.  In the data frame, transform the variable charges by seting
#         insurance$charges = log(charges). Do not transform it
#         outside of the data frame.
insurance$charges <- log(insurance$charges)
#     c.  Using the data set from 1.b, use the model.matrix() function
#         to create another data set that uses dummy variables in place
#         of categorical variables. Verify that the first column only has
#         ones (1) as values, and then discard the column only after
#         verifying it has only ones as values.
insurance.matrix <- as.data.frame(model.matrix(~ age + sex + bmi + children + smoker + region + charges, data = insurance))
unique(insurance.matrix[,1])
insurance.matrix <- insurance.matrix[,-1]
#     d.  Use the sample() function with set.seed equal to 1 to generate
#         row indexes for your training and tests sets, with 2/3 of the
#         row indexes for your training set and 1/3 for your test set. Do
#         not use any method other than the sample() function for
#         splitting your data.
set.seed(1)
train.index <- sample(1:nrow(insurance), (2/3)*nrow(insurance))
#     e.  Create a training and test data set from the data set created in
#         1.b using the training and test row indexes created in 1.d.
#         Unless otherwise stated, only use the training and test
#         data sets created in this step.
test <- insurance[-train.index,]
train <- insurance[train.index,]
#     f.  Create a training and test data set from data set created in 1.c
#         using the training and test row indexes created in 1.d
model.test <- insurance.matrix[-train.index,]
model.train <- insurance.matrix[train.index,]

#################################################
# 2.  Build a multiple linear regression model. #
#################################################

#     a.  Perform multiple linear regression with charges as the
#         response and the predictors are age, sex, bmi, children,
#         smoker, and region. Print out the results using the
#         summary() function. Use the full data set created in step
#         1.b to train your model.
fit <- lm(charges ~ age + sex + bmi + children + smoker + region, data = train)
summary(fit)
#     b.  Is there a relationship between the predictors and the
#         response?
# ANSWER: Yes. The p=value is less than the cutoff value.
#     c.  Does sex have a statistically significant relationship to the
#         response?
# ANSWER: Yes. Sex has a p-value of 0.001650, which is below the cutoff value.
#     d.  Perform best subset selection using the stepAIC() function
#         from the MASS library, choose best model based on AIC. For
#         the "direction" parameter in the stepAIC() method, set
#         direciton="backward"
require(MASS)
fit.best <- stepAIC(fit, direction = "backward")
summary(fit.best)
#     e.  Compute the test error of the best model in #3d based on AIC
#         using LOOCV using trainControl() and train() from the caret
#         library. Report the MSE by squaring the reported RMSE.
require(caret)
train_control <- trainControl(method = "LOOCV")
model <- train(charges ~ age + sex + bmi + children + smoker + region, data = insurance, trControl=train_control, method = "lm")
print(model)
(0.4458979)^2
# THE MSE IS : 0.1988249
#     f.  Calculate the test error of the best model in #3d based on AIC
#         using 10-fold Cross-Validation. Use train and trainControl
#         from the caret library. Refer to model selected in #3d based
#         on AIC. Report the MSE.
set.seed(1)
train_control <- trainControl(method="cv", number = 10)
model <- train(charges ~ age + sex + bmi + children + smoker + region, data = insurance,
               trControl=train_control, method="lm")
print(model)
cv.MSE <- (0.4448879)^2
cv.MSE
# THE MSE IS : 0.1979252
#     g.  Calculate and report the test MSE using the best model from 
#         2.d and the test data set from step 1.e.
pred <- predict(fit.best, newdata=test)

MSE.lm <- mean((test[,"charges"] - pred)^2)

MSE.lm
#     h.  Compare the test MSE calculated in step 2.f using 10-fold
#         cross-validation with the test MSE calculated in step 2.g.
#         How similar are they?
cv.MSE
MSE.lm
MSE.lm-cv.MSE
# ANSWER : The MSE I calculated is a bit bigger.
######################################
# 3.  Build a regression tree model. #
######################################

#     a.  Build a regression tree model using function tree(), where
#         charges is the response and the predictors are age, sex, bmi,
#         children, smoker, and region.
require(tree)
fit.tree <- tree(charges ~ age + sex + bmi + children + smoker + region, data = train)
summary(fit.tree)
#     b.  Find the optimal tree by using cross-validation and display
#         the results in a graphic. Report the best size.
set.seed(1)
cv.tree.results <- cv.tree(fit.tree)

plot(cv.tree.results$size, cv.tree.results$dev, type="b",
     xlab="Number of terminal nodes", ylab="Deviance",
     main="Cross-validation for Choice of Tree Complexity")
# ANSWER: the best size would be 3, according to the graph.

#     c.  Justify the number you picked for the optimal tree with
#         regard to the principle of variance-bias trade-off.

# ANSWER: the way I balanced bias and variance was by picking a number
# of nodes that had a low deviance, while keeping the complexity of the
# model on the lower side. I picked the model by using the prior visualization.

#     d.  Prune the tree using the optinal size found in 3.b
set.seed(1)
prune.tree.model <- prune.tree(fit.tree, best = 3)
#     e.  Plot the best tree model and give labels.
plot(prune.tree.model)
text(prune.tree.model, pretty=0)
#     f.  Calculate the test MSE for the best model.
pred <- predict(prune.tree.model, newdata=test)

MSE.tree <- mean((test[,"charges"] - pred)^2)

MSE.tree
####################################
# 4.  Build a random forest model. #
####################################

#     a.  Build a random forest model using function randomForest(),
#         where charges is the response and the predictors are age, sex,
#         bmi, children, smoker, and region.
require(randomForest)
set.seed(1)
fit.rf <- randomForest(charges ~ age + sex + bmi + children + smoker + region, data = train, importance = TRUE)
#     b.  Compute the test error using the test data set.
pred.rf <- predict(fit.rf, newdata=test, type="response")
MSE.rf  <- mean((test[,"charges"] - pred.rf)^2)

MSE.rf
#     c.  Extract variable importance measure using the importance()
#         function.
importance(fit.rf)
#     d.  Plot the variable importance using the function, varImpPlot().
#         Which are the top 3 important predictors in this model?
varImpPlot(fit.rf)
############################################
# 5.  Build a support vector machine model #
############################################

#     a.  The response is charges and the predictors are age, sex, bmi,
#         children, smoker, and region. Please use the svm() function
#         with radial kernel and gamma=5 and cost = 50.
require(e1071)
f <- formula(charges ~ age + sex + bmi + children + smoker + region)
set.seed(1)
fit.svm <- svm(f, data = train, kernel = "radial", gamma=5, cost=50)
summary(fit.svm)
#     b.  Perform a grid search to find the best model with potential
#         cost: 1, 10, 50, 100 and potential gamma: 1,3 and 5 and
#         potential kernel: "linear","polynomial","radial" and
#         "sigmoid". And use the training set created in step 1.e.
set.seed(1)
insurance.tune <- tune(svm, f, data=train,
                  ranges=list(
                    cost=c(1,10,50,100),
                    gamma=c(1,3,5),
                    kernel=c("linear", "radial", "sigmoid")))
#     c.  Print out the model results. What are the best model
#         parameters?
summary(insurance.tune)
#     d.  Forecast charges using the test dataset and the best model
#         found in c).
insurance.svm.pred <- predict(insurance.tune$best.model, newdata=test)
#     e.  Compute the MSE (Mean Squared Error) on the test data.
MSE.svm <- mean((test[,"charges"] - insurance.svm.pred)^2)
MSE.svm
#############################################
# 6.  Perform the k-means cluster analysis. #
#############################################

#     a.  Use the training data set created in step 1.f and standardize
#         the inputs using the scale() function.
scaled.data <- scale(model.train[,c(1:8)])
#     b.  Convert the standardized inputs to a data frame using the
#         as.data.frame() function.
model.scaled <- as.data.frame(scaled.data)
#     c.  Determine the optimal number of clusters, and use the
#         gap_stat method and set iter.max=20. Justify your answer.
#         It may take longer running time since it uses a large dataset.
require(cluster)
require(factoextra)
set.seed(1)
fviz_nbclust(model.scaled, kmeans, method = "gap_stat", iter.max = 20)
# Answer: 3

#     d.  Perform k-means clustering using the optimal number of
#         clusters found in step 6.c. Set parameter nstart = 25
km.res <- kmeans(model.scaled, 3, iter.max = 20, nstart = 25)
#     e.  Visualize the clusters in different colors, setting parameter
#         geom="point"
fviz_cluster(km.res, data = model.scaled, geom = "point")

######################################
# 7.  Build a neural networks model. #
######################################

#     a.  Using the training data set created in step 1.f, create a 
#         neural network model where the response is charges and the
#         predictors are age, sexmale, bmi, children, smokeryes, 
#         regionnorthwest, regionsoutheast, and regionsouthwest.
#         Please use 1 hidden layer with 1 neuron. Do not scale
#         the data.
require(neuralnet)
set.seed(1)
fit.nn <- neuralnet(charges ~ age + sexmale + bmi + children + smokeryes + regionnorthwest + regionsoutheast +
                      regionsouthwest, data = model.train, hidden = 1)
#     b.  Plot the neural network.
plot(fit.nn)
#     c.  Forecast the charges in the test dataset.
forecast <- compute(fit.nn, model.test[,-9])
#     d.  Compute test error (MSE).
observ.test <- model.test$charges

MSE.nn <- mean((observ.test - forecast$net.result)^2)
MSE.nn
################################
# 8.  Putting it all together. #
################################

#     a.  For predicting insurance charges, your supervisor asks you to
#         choose the best model among the multiple regression,
#         regression tree, random forest, support vector machine, and
#         neural network models. Compare the test MSEs of the models
#         generated in steps 2.g, 3.f, 4.b, 5.e, and 7.d. Display the names
#         for these types of these models, using these labels:
#         "Multiple Linear Regression", "Regression Tree", "Random Forest", 
#         "Support Vector Machine", and "Neural Network" and their
#         corresponding test MSEs in a data.frame. Label the column in your
#         data frame with the labels as "Model.Type", and label the column
#         with the test MSEs as "Test.MSE" and round the data in this
#         column to 4 decimal places. Present the formatted data to your
#         supervisor and recommend which model is best and why.
MSE.frame <- data.frame(Model.Type=c("Multiple Linear Regression", "Regression Tree", "Random Forest", "Support Vector Machine", "Neural Network"),
                        Test.MSE=c(MSE.lm, MSE.tree, MSE.rf, MSE.svm, MSE.nn))
MSE.frame$Test.MSE <- round(MSE.frame$Test.MSE, 4)
MSE.frame


# ANSWER: The model with the lowest Mean Squared Error (and therefore the best model)
# is the random forest with an MSE of 0.1656. The one with the highest MSE is
# the neural network, with an MSE of 0.8213.

#     b.  Another supervisor from the sales department has requested
#         your help to create a predictive model that his sales
#         representatives can use to explain to clients what the potential
#         costs could be for different kinds of customers, and they need
#         an easy and visual way of explaining it. What model would
#         you recommend, and what are the benefits and disadvantages
#         of your recommended model compared to other models?

# The model I would recommend would be the regression tree, as it shows the prediction
# based upon a couple of variables in a very visual way and in a way that would be
# easy to understand for a layperson who does not know anything about neural networks
# or support vector machines.

# The disadvantages would be that it does have a tendency to oversimplify results

#     c.  The supervisor from the sales department likes your regression
#         tree model. But she says that the sales people say the numbers
#         in it are way too low and suggests that maybe the numbers
#         on the leaf nodes predicting charges are log transformations
#         of the actual charges. You realize that in step 1.b of this
#         project that you had indeed transformed charges using the log
#         function. And now you realize that you need to reverse the
#         transformation in your final output. The solution you have
#         is to reverse the log transformation of the variables in 
#         the regression tree model you created and redisplay the result.
#         Follow these steps:
#
#         i.   Copy your pruned tree model to a new variable.

copy.prune <- prune.tree.model

#         ii.  In your new variable, find the data.frame named
#              "frame" and reverse the log transformation on the
#              data.frame column yval using the exp() function.
#              (If the copy of your pruned tree model is named 
#              copy_of_my_pruned_tree, then the data frame is
#              accessed as copy_of_my_pruned_tree$frame, and it
#              works just like a normal data frame.).
copy.prune$frame$yval <- exp(copy.prune$frame$yval)

#         iii. After you reverse the log transform on the yval
#              column, then replot the tree with labels.
plot(copy.prune)
text(copy.prune, pretty=0)
