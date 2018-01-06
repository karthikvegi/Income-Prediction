#--------------------------------------------
# Project:
# Author: Karthik Vegi
#--------------------------------------------
# The code depends on the below packages. Install them if missing
# install.packages("e1071")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("summarytools")

# Required libraries
library("e1071")
library("rpart")
library(rpart.plot)

# Loading the data.. placed one level outside
census.train <- read.table("../adult.data", sep=",")
census.test <- read.table("../adult.test", sep=",")

# Adding variable names
names(census.train) <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                         "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                         "weekly-hours", "native-country", "income")

# Check the frequency of the classes
freq<-as.data.frame(table(census.train$income))
cat("Class Frequencies for training set...\n")
freq$proportion <- (freq$Freq/nrow(census.train))*100
print(freq)
print(dfSummary(census.train, style = "grid"))

# Feature selection: excluding fnlwgt and education-num
census.train <- census.train[-c(3,5)]

# Export dataset for visualization
write.csv(census.train, 'census_train.csv', row.names = T)
write.csv(census.test, 'census_test.csv', row.names = T)

# (a) Applying Naive Bayes classifier on training data
naive.model <- naiveBayes(income ~ .-income, data = census.train)

# Use the training data to predict
train.pred <- predict(naive.model, census.train)

# Create confusion mattix for training data
cat("\nNaive Bayes: Confusion matrix for training set..")
print(table(train.pred, census.train$income))

# Accuracy on training
cat("\nNaive Bayes: Accuracy of classifier on the training set is..")
print((sum(train.pred==census.train$income)/nrow(census.train))*100)

# Loading the test data
names(census.test) <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                        "weekly-hours", "native-country", "income")

# Feature selection: excluding fnlwgt and education-num
census.test <- census.test[-c(3,5)]

# Use the test data to predict
naive.pred <- predict(naive.model, census.test)

# Create confusion matrix for test data
cat("\nNaive Bayes: Confusion matrix for test set..")
print(table(naive.pred, census.test$income))
plot(naive.pred, census.test$income, main=" Naive bayes")

# Accuracy on test set
cat("\nNaive Bayes: Accuracy of classifier on the test set is..")
naive.acc <- (sum(naive.pred==census.test$income)/nrow(census.test))*100
print(naive.acc)

# (b): Applying Decision Tree Classification on training set
train.model <- rpart(income ~ ., data= census.train, method = "class")
printcp(train.model)
plotcp(train.model)
prp(train.model)

# prune the tree
train.model1 <- prune(train.model, cp=0.010000)

# Create confusion matrix for training data
cat("\nDecision Tree: Confusion matrix for training set..")
train.pred <- predict(train.model, data=census.train, type="class")
print(table(train.pred, census.train$income))

# Accuracy on training set
cat("\nDecision Tree: Accuracy of classifier on the training set is..")
print((sum(train.pred==census.train$income)/nrow(census.train))*100)

# Applying Decision Tree Classification on test set
tree.pred <- predict(train.model, census.test, type="class")

# Create confusion matrix for training data
cat("\nDecision Tree: Confusion matrix for test set..")
print(table(tree.pred, census.test$income))

# Accuracy on test set
cat("\nDecision Tree: Accuracy of classifier on the test set is..")
tree.acc <- (sum(tree.pred==census.test$income)/nrow(census.test))*100
print(tree.acc)
plot(tree.pred, census.test$income, main="Decision Tree")

# Comparision of accuracy
for(i in 1:2) {
  if(i==1) {
    plot(i, naive.acc, type="h", xlim=c(1,5), ylim=c(1,100), xlab=" Classifier", ylab="Accuracy",
         main ="Comparision of Naive Bayes(Red) Vs Decision Tree(Blue)", col="red")
  } else {
    points(i, tree.acc, type="h", col="blue")
  }
}
