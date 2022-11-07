
# Load the Concrete data

summary(concrete)

# custom normalization function
normalize <- function(x) { 
     return((x - min(x)) / (max(x) - min(x)))
     }

# Apply normalization function to entire data
concrete_norm <- as.data.frame(lapply(concrete, normalize))

summary(concrete_norm)
# create training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

# Train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# ANN with only a single hidden neuron
concrete_model <- neuralnet(formula = strength ~ cement + slag + ash + water + superplastic 
                            + coarseagg + fineagg + age, data = concrete_train)
# visualize the network topology
plot(concrete_model)

## Evaluating model performance
model_results <- compute(concrete_model, concrete_test[1:8])

# Obtain predicted strength values
predicted_strength <- model_results$net.result

# Examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Improving model performance
# A more complex neural network topology with 5 hidden neurons
concrete_model2 <- neuralnet(strength ~ cement + slag +
                                  ash + water + superplastic + 
                                  coarseagg + fineagg + age,
                             data = concrete_train, hidden = 5)

# plot the network
plot(concrete_model2)

# Evaluate the results
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)
