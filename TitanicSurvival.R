# Set working directory
setwd("F:/Documents/Programming/R/TitanicSurvivalPredictions")

# Import the training set: train
train_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train <- read.csv(train_url)
write.csv(train, file = "train.csv", row.names = FALSE)
  
# Import the testing set: test
test_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test <- read.csv(test_url)
write.csv(test, file = "test.csv", row.names = FALSE)

# View metadata about the datasets
str(train)
str(test)

# Survival rates in absolute numbers
table(train$Survived)

# Survival rates in proportions
prop.table(table(train$Survived))
  
# Two-way comparison: Sex and Survived
table(train$Sex, train$Survived)

# Two-way comparison: row-wise proportions
prop.table(table(train$Sex, train$Survived), 1)

# Create the column child, and indicate whether child or no child
train$Child <- NA
train$Child[train$Age < 18] <- 1
train$Child[train$Age >= 18] <- 0

# Two-way comparison
prop.table(table(train$Child, train$Survived), 1)

# Copy of test
test_one <- test

# Initialize a Survived column to 0
test_one$Survived <- 0

# Super simple prediction, if a passenger is female they survive
# Set Survived to 1 if Sex equals "female"
test_one$Survived[test_one$Sex == "female"] <- 1

#################################################
# Better way to determine variables to split on #
#################################################

# Load in the R package
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Build the decision tree
my_tree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")

# Visualize the decision tree using plot() and text()
plot(my_tree)
text(my_tree)

# Plot fancy tree
fancyRpartPlot(my_tree)

# Make predictions on the test set
my_prediction <- predict(my_tree, newdata = test, type = "class")

# Finish the data.frame() call
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Use nrow() on my_solution
nrow(my_solution)

# Finish the write.csv() call
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

# Result: 0.78469

# Build second tree
# cp determines when the splitting up of the decision tree stops
# minsplit determines the minimum amount of observations in a leaf of the tree
my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                     data = train, method = "class", control = rpart.control(minsplit = 50, cp = 0))

# Visualize the new tree
fancyRpartPlot(my_tree_two)

# Use feature engineering to create a new variable representing the size of a family
train_two <- train
train_two$family_size <- train_two$SibSp + train_two$Parch + 1

# Finish the command
my_tree_three <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size,
                      data = train_two, method = "class")

# Visualize your new decision tree
fancyRpartPlot(my_tree_three)

# Plot did not change, family size did not have a valuable enough affect