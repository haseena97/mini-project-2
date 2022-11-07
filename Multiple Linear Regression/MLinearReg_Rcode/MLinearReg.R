library(readr)
Cars <- read.csv(file.choose())
attach(Cars)
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
summary(Cars)

plot(HP,MPG)#scatter plot between Hp&MPG as HP inc MPG dec

cor(HP,MPG)#strength of the relation ship bw (HP,MPG)

plot(VOL,MPG)

# 7. Find the correlation between Output (MPG) & inputs (HP, VOL, SP, WT) - SCATTER DIAGRAM
pairs(Cars)

# 8. Correlation coefficient Matrix - Strength & Direction of correlation
cor(Cars)

# The Linear Model of interest
model.car <- lm(MPG ~ VOL + HP + SP + WT)
summary(model.car)

install.packages("car")
library(car)

vif(model.car) # variance inflation factor

model2 <- lm(MPG ~ VOL + HP + SP)
summary(model2)
vif(model2)


# Data Partitioning
n <-  nrow(Cars) #will give whole number of observations
n1 <-  (n* 0.7) #we want 70% of the obs as train dataset
n2 <-  n - n1 #rest of the observations which is 30% as test dataset
train <-  sample(1:n, n1)#taking a random sample of size n1 from the n numbers as train dataset
traindata <- cars[train,]#the corresponding obs
test <-  Cars[-train, ]#the observations which is not present in the train dataset is test dataset

model <- lm(MPG~VOL+HP+SP, data <- Cars[train, ])
summary(model)
?lm

pred <-  predict(model2, newdata = test)
actual <-  test$MPG
error <-  actual - pred

test.rmse <-  sqrt(mean(error**2))
test.rmse

train.rmse <-  sqrt(mean(model$residuals**2))
train.rmse
