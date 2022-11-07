# Load the Dataset

claimants <- read.csv(file.choose()) # Choose the claimants Data set
View (claimants)
colnames(claimants) # give column names
claimants <- claimants[, -1] # Removing the first column which is is an Index
# split into train and test
claimants_train <- claimants[1:1000, ]
claimants_test <- claimants[1001:1340, ]
# train data into logistic model (binomial)
model <- glm(ATTORNEY~., data = claimants_train, family = "binomial")
summary(model)

# Prediction on Test data 
prob_test <- predict(model, claimants_test, type="response")
prob_test

# Confusion matrix and considering the threshold value as 0.5 
confusion_test <- table(prob_test>0.5, claimants_test$ATTORNEY)
confusion_test

# Model Accuracy 
Accuracy_test <- sum(diag(confusion_test)) / sum(confusion_test)
Accuracy_test


# Prediction on Train data 
prob_train <- predict(model, claimants_train, type="response")

# Confusion matrix and considering the threshold value as 0.5 
confusion_train <- table(prob_train > 0.5, claimants_train$ATTORNEY)
confusion_train

# Model Accuracy 
Accuracy_train <- sum(diag(confusion_train)) / sum(confusion_train)
Accuracy_train
