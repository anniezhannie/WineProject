###################################
# Capstone Project 2: Wine Quality
###################################


### Import and View Dataset ###

#import data from csv file
data <- read.csv2("winequality-red.csv", stringsAsFactors=FALSE, header=TRUE)

#take a look at the data
head(data)
summary(data)
str(data)


### Data Pre-Processing ###

# convert variables to numeric
data$fixed.acidity <- as.numeric(data$fixed.acidity)
data$volatile.acidity <- as.numeric(data$volatile.acidity)
data$citric.acid <- as.numeric(data$citric.acid)
data$residual.sugar <- as.numeric(data$residual.sugar)
data$chlorides <- as.numeric(data$chlorides)
data$free.sulfur.dioxide <- as.numeric(data$free.sulfur.dioxide)
data$total.sulfur.dioxide <- as.numeric(data$total.sulfur.dioxide)
data$density <- as.numeric(data$density)
data$pH <- as.numeric(data$pH)
data$sulphates <- as.numeric(data$sulphates)
data$alcohol <- as.numeric(data$alcohol)

# Load libraries needed
library(tidyverse)
library(caret)
library(data.table)
set.seed(1)

# Validation set will be 10% of MovieLens data
test_index <- createDataPartition(y = data$quality, times = 1, p = 0.2, list = FALSE)
training <- data[-test_index,]
test <- data[test_index,]

# can also remove quality score from validation set
test_withoutquality <- test %>% select(-quality)


### Data Visualization ###

# Plot of distribution of quality scores
hist(data$quality)

library(graphics)

par(mfrow=c(4,3), mar = c(4, 4, 1, 2))

# Plot of Quality vs. Fixed Acidity
plot(quality~fixed.acidity,data=data,
     main="Quality Score vs. Fixed Acidity",
     col="steelblue3")

# Plot of Quality vs. Volatile Acidity
plot(quality~fixed.acidity,data=data,
     main="Quality Score vs. Volatile Acidity",
     col="steelblue3")

# Plot of Quality vs. Citric Acid
plot(quality~citric.acid,data=data,
     main="Quality Score vs. Citric Acid",
     col="steelblue3")

# Plot of Quality vs. Residual Sugar
plot(quality~residual.sugar,data=data,
     main="Quality Score vs. Residual Sugar",
     col="steelblue3")

# Plot of Quality vs. Chlorides
plot(quality~chlorides,data=data,
     main="Quality Score vs. Chlorides",
     col="steelblue3")

# Plot of Quality vs. free.sulfur.dioxide
plot(quality~free.sulfur.dioxide,data=data,
     main="Quality Score vs. Free SO2",
     col="steelblue3")

# Plot of Quality vs. total.sulfur.dioxide
plot(quality~total.sulfur.dioxide,data=data,
     main="Quality Score vs. Total SO2",
     col="steelblue3")

# Plot of Quality vs. Density
plot(quality~density,data=data,
     main="Quality Score vs. Density",
     col="steelblue3")

# Plot of Quality vs. pH
plot(quality~pH,data=data,
     main="Quality Score vs. pH",
     col="steelblue3")

# Plot of Quality vs. Sulphates
plot(quality~sulphates,data=data,
     main="Quality Score vs. Sulphates",
     col="steelblue3")

# Plot of Quality vs. Alcohol
plot(quality~alcohol,data=data,
     main="Quality Score vs. Alcohol",
     col="steelblue3")


### Variable Exploration ###

# plot of all variables against each other
plot(data)

# Calculate correlations table (removed categorical variable)
library(corrplot)
correlations <- cor(data)
correlations

# Display correlation plot
corrplot(correlations, order="hclust")


### ANALYSIS 1. Multivariate Regression Models ###

### First Regression Model ###

# Build first basic multivariate regression model with all predictors and data
model1 <- lm(quality~., data=training)
summary(model1)

# Plot of Cook's Distance
cooks <- cooks.distance(model1)
plot(cooks,type="h", lwd=3, col="steelblue3", ylab="Cook's Distance")
abline(1,0,col="steelblue1")

# Identify possible outliers
outliers <- as.numeric(names(cooks)[(cooks > 0.04)])
outliers

# Remove possible outliers
training2 <- training[-c(152, 653, 1236),]

### Second Regression Model Without Outliers ###

# Build second multiple linear regression model without possible outliers
model2<-lm(quality~., data=training2)
summary(model2)

library(car)

# Display VIF
vif(model2)

# Compute Threshold for VIF
1/(1-summary(model2)$r.squared)

### Third Regression Model Reduced Number of Predictor Variables ###

# Build third multiple linear regression model without possible outliers and redundant predictor variables
model3<-lm(quality~volatile.acidity+chlorides+total.sulfur.dioxide+sulphates+alcohol, data=training2)
summary(model3)

# Display VIF
vif(model3)

# Compute VIF Threshold
1/(1-summary(model3)$r.squared)

# Find standardized residuals
resids <- rstandard(model3)

par(mfrow=c(1,2))

plot(model2$fitted.values, resids, main="Residuals vs Fitted",
     xlab="Fitted Values", ylab="Residuals")
abline(h=0, col="red")

### Analysis of Multivariate Regression Models ###

# Find standardized residuals
resids <- rstandard(model3)

plot(model2$fitted.values, resids, main="Residuals vs Fitted",
     xlab="Fitted Values", ylab="Residuals")
abline(h=0, col="red")

# Look at residuals plots
par(mfrow=c(1,2))

hist(resids, col="steelblue3", nclass=15, main="Histogram of Residuals")
qqPlot(resids, main="Normal QQ Probability Plot")
qqline(resids, col="steelblue3")


### ANALYSIS 2. K-Nearest Neighbours Models ###

library(kknn)
library(tidyverse)
library(FNN)

# Put response in its own object
quality_response <- data %>% select(quality)

# Define response in each subset
response_train <- quality_response[-test_index, ]
response_test <- quality_response[test_index, ]

# define function to calculate RMSE
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

# Check different values of k

# For k=1
modelknn <- knn.reg(training, test=NULL, response_train, k = 1)
predictions <- as.integer(modelknn$pred)
rmse(response_train, predictions)

# For k=3
modelknn <- knn.reg(training, test=NULL, response_train, k = 3)
predictions <- as.integer(modelknn$pred)
rmse(response_train, predictions)

# For k=5
modelknn <- knn.reg(training, test=NULL, response_train, k = 5)
predictions <- as.integer(modelknn$pred)
rmse(response_train, predictions)

# For k=10
modelknn <- knn.reg(training, test=NULL, response_train, k = 10)
predictions <- as.integer(modelknn$pred)
rmse(response_train, predictions)

# For k=15
modelknn <- knn.reg(training, test=NULL, response_train, k = 15)
predictions <- as.integer(modelknn$pred)
rmse(response_train, predictions)

# For k=30
modelknn <- knn.reg(training, test=NULL, response_train, k = 30)
predictions <- as.integer(modelknn$pred)
rmse(response_train, predictions)

### Our Final KNN Model ###

KNN_model <- knn.reg(training, test=NULL, response_train, k = 1)
head(KNN_model)


### RESULTS ###

### Multivariate Regression Model RMSE Evaluation ###

# KNN model used to predict test dataset
predicted_reg <- predict(model3, test)

# Calculate RMSE against test dataset
reg_rmse <- rmse(response_test, predicted_reg)
reg_rmse

#Compare RMSE of Model 3 to other regression models

# Model 1
predicted_reg1 <- predict(model1, test)
reg_rmse1 <- rmse(response_test, predicted_reg1)
reg_rmse1

# Model 2
predicted_reg2 <- predict(model2, test)
reg_rmse2 <- rmse(response_test, predicted_reg2)
reg_rmse2


### K Nearest Neighbours Model RMSE Evaluation ###

# KNN model used to predict test dataset
KNN_model2 <- knn.reg(train=training, test=test, y=response_train, k = 1)
predicted_knn <- as.integer(KNN_model2$pred)

# Calculate RMSE against test dataset
knn_rmse <- rmse(response_test, predicted_knn)
knn_rmse


### Create Table of RMSE ###

# Create RMSE table of RMSE results, starting with regression model
rmsetable <- data.frame(Model="Multivariate Regression Model", RMSE = reg_rmse)

# Add KNN model RMSE results
rmsetable <- bind_rows(rmsetable, data_frame(Model="K Nearest Neighbours Model", RMSE = knn_rmse))

# Load RMSE table
rmsetable %>% knitr::kable()

#######################################
# Thanks for reading my submission! :)
#######################################
