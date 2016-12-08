# Import the training set: train
train_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train <- read.csv(train_url)
  
# Import the testing set: test
test_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test <- read.csv(test_url)

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


