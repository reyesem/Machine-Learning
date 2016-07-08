################################################################################
# Program: run_analysis
# Project: Data Science Course: Machine Learning Project
# Description: Builds a model for predicting the response of interest, estimates
#              the out of sample error, and predicts for 20 unknown cases.
#
# Author: Eric Reyes
# Date: Summer 2016
# Modified:
#
# Notes:


################################################################################
# Header

# Load Packages
pkgs <- c("caret", "plyr", "dplyr", "gbm", "GGally", "ggplot2", "gridExtra", 
          "rattle")
for(pkg in pkgs) library(pkg, character.only=TRUE)

# Specify Directory
dir <- "//MyDocs/MyDocs/reyesem/Documents/DataScience/"




################################################################################
# Obtain Data
#  There is both a test and training dataset. The test dataset contains the
#  features for a group of subjects for which the class is unknown. We are to
#  classify each of these 20 individuals.
#
#  We split the data into three distinct datasets.
#   1. df.pred - This is synonymous with the original "testing" dataset and
#       contains observations which we need to classify.
#   2. df.train - A subset of the original "training" dataset that will be
#       used for building the proposed model.
#   3. df.test - A subset of the original "training" dataset that will be used
#       to assess our final model's performance. This will not be examined 
#       further until the end of the analysis.
#
#  When splitting into test/training data, we attempt to keep some of the 
#  structure of the data. There are only 6 subjects, each of which underwent
#  5 types of lifts (the class variable of interest). When sampling, we sample
#  from within each subject/classification type.

# Read Training Data
pmltrain <- read.csv(paste(dir, "Data/PML HAR Dataset/pmltraining.csv", sep=""),
                     header=TRUE, stringsAsFactors=FALSE)


# Read Testing Data
pmltest <- read.csv(paste(dir, "Data/PML HAR Dataset/pmltesting.csv", sep=""),
                    header=TRUE, stringsAsFactors=FALSE)


### Split Into Prediction/Training/Testing Data

# Prediction Dataset
df.pred <- pmltest

# Training/Testing Dataset
#  The seed controls the random splitting and the method ensures the data
#  are subsampled within the class/subject combinations.
set.seed(20160706)
inTrain <- createDataPartition(interaction(pmltrain$classe,
                                           pmltrain$user_name),
                               times=1,
                               p=0.7,
                               list=FALSE)

df.train <- slice(pmltrain, inTrain)
df.test <- slice(pmltrain, -inTrain)



################################################################################
# Exploratory Analysis
#  Examining the features available (not missing) in the prediction dataset
#  informs our choice of potential predictors in the training dataset. Data
#  were collected using three devices (gyroscope, accelerometer, magnet) in
#  each of three directions (x, y, z) in each of several locations (belt,
#  arm, dumbell, forearm).  Only the composites will be used in the model
#  building (amplitude, total acceleration, etc) instead of the individual
#  components. In addition, we will not use information regarding the timestamp
#  of the lifts.  While this may be extremely predictive, it will not be 
#  useful for future predictions. Similar arguments can be made regarding the
#  indicator of being in a new window as well as the username. This assumes that
#  the goal of the model is future prediction for any subject; not that a model
#  is constructed for an individual by "training" the device first.
#
#  First, we simply examine several plots to explore various relationships
#  within the data.

# ### Verification of Timestamp/Window Role
# # This plot verfies the belief that lifts of the same type were done
# # sequentially. Therefore, the timestamp will unqiuely identify which type of
# # lift was being conducted.
# ggplot(data=df.train, 
#        mapping=aes(x=raw_timestamp_part_1,
#                    y=num_window,
#                    colour=classe)) +
#   geom_point() +
#   labs(x="Timestamp (part 1)", y="Window Number", colour="Lift Type")
#   facet_wrap(~user_name, scales="free") +
#   theme_bw() +
#   theme(legend.position="bottom")
# 
# 
# ### Belt Measurements
# # Scatterplot Matrix
# # This allows us to view both the marginal as well as any conditional 
# # relationships within the data.
# #
# # The graphics suggest that the classes are not easily identified.
# ggpairs(df.train, mapping=aes(colour=classe),
#         columns=c("roll_belt", "pitch_belt", "yaw_belt",
#                   "total_accel_belt"),
#         lower=list(continuous=wrap("points", alpha=0.5)),
#         diag=list(continuous=wrap("densityDiag", alpha=0.5)))
# 
# 
# ### Arm Measurements
# # Scatterplot Matrix
# # This allows us to view both the marginal as well as any conditional 
# # relationships within the data.
# #
# # The graphics suggest that the classes are not easily identified.
# ggpairs(df.train, mapping=aes(colour=classe),
#         columns=c("roll_arm", "pitch_arm", "yaw_arm",
#                   "total_accel_arm"),
#         lower=list(continuous=wrap("points", alpha=0.5)),
#         diag=list(continuous=wrap("densityDiag", alpha=0.5)))
# 
# 
# ### Dumbbell Measurements
# # Scatterplot Matrix
# # This allows us to view both the marginal as well as any conditional 
# # relationships within the data.
# #
# # The graphics suggest that the classes are not easily identified.
# ggpairs(df.train, mapping=aes(colour=classe),
#         columns=c("roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell",
#                   "total_accel_dumbbell"),
#         lower=list(continuous=wrap("points", alpha=0.5)),
#         diag=list(continuous=wrap("densityDiag", alpha=0.5)))
# 
# 
# ### Forearm Measurements
# # Scatterplot Matrix
# # This allows us to view both the marginal as well as any conditional 
# # relationships within the data.
# #
# # The graphics suggest that the classes are not easily identified.
# ggpairs(df.train, mapping=aes(colour=classe),
#         columns=c("roll_forearm", "pitch_forearm", "yaw_forearm",
#                   "total_accel_forearm"),
#         lower=list(continuous=wrap("points", alpha=0.5)),
#         diag=list(continuous=wrap("densityDiag", alpha=0.5)))




################################################################################
# Model Building
#  Using the features explored previously, we build a model. Since we did not
#  observe any major linear trends in the exploratory analysis, we focus on
#  a nonlinear model - a classification tree.  To improve interpretability,
#  we are not going to use a random forest approach because one goal of the
#  study was to provide user feedback, which might be improved if the model is
#  easy to interpret.
#
#  The tuning parameter for the classification tree is cp, the complexity 
#  parameter which eliminates splits which dos not decrease the overall lack of
#  fit by a factor of cp  It is tuned using 1 replication of 10-fold cross
#  validation. 
#
#  A grid of values is used when searching for optimal value of cp. 25 different
#  possibilities are examined.
#
#  We also considered a boosted CART model via gradient boosting. The tuning
#  parameters in this case include the number of trees to include, the
#  interaction depth, the shrinkage term and the number of minimum observations
#  in a node. These were tuned using 1 replication of 10-fold cross validation.
#
#  A grid of values is used when searching for optimal values of these 
#  parameters. 10 different combinations were considered.
#
#  Note that gradient boosting is somewhat computationally intensive.

# Specify controls
set.seed(20160707)
controls <- trainControl(method="repeatedcv",
                         number=10,
                         repeats=1)

# Train model
#  Recursive tree fit using the rpart package.
fit.rpart <- train(classe ~ 
                     roll_belt + pitch_belt + yaw_belt + total_accel_belt +
                     roll_arm + pitch_arm + yaw_arm + total_accel_arm +
                     roll_dumbbell + pitch_dumbbell + yaw_dumbbell + total_accel_dumbbell +
                     roll_forearm + pitch_forearm + yaw_forearm + total_accel_forearm,
                   data=df.train, method="rpart", trControl=controls,
                   tuneLength=25)

fit.gbm <- train(classe ~
                   roll_belt + pitch_belt + yaw_belt + total_accel_belt +
                   roll_arm + pitch_arm + yaw_arm + total_accel_arm +
                   roll_dumbbell + pitch_dumbbell + yaw_dumbbell + total_accel_dumbbell +
                   roll_forearm + pitch_forearm + yaw_forearm + total_accel_forearm,
                 data=df.train, method="gbm", trControl=controls,
                 tuneLength=10, verbose=FALSE)


# Visual Representation
#  Graphically illustrate the classification tree.
fancyRpartPlot(fit.rpart$finalModel, main="", sub="")


# Assess Model in Training Set
#  This will be a biased view of how well the model performs.
confusionMatrix(predict(fit.rpart, newdata=df.train), df.train$classe)
confusionMatrix(predict(fit.gbm, newdata=df.train), df.train$classe)




################################################################################
# Model Assessment
#  The model is assessed in the test set to get an idea of the out-of-sample
#  accuracy of the model.

confusionMatrix(predict(fit.rpart, newdata=df.test), df.test$classe)
confusionMatrix(predict(fit.gbm, newdata=df.test), df.test$classe)




################################################################################
# Future Predictions
#  For the unknown cases, compute their predictions.

data_frame(CART = predict(fit.rpart, newdata=df.pred),
           GBM = predict(fit.gbm, newdata=df.pred))




################################################################################
# Exact Predictions (Cheating)
#  Using the timestamp, we should be able to perfectly classify the subjects.
#  The theory is tested and used to form perfect predictions.

# # Fit Model
# set.seed(20160707)
# fit.cheat <- train(classe ~ raw_timestamp_part_1 + user_name,
#                    data=df.train, method="rpart", trControl=controls, 
#                    tuneLength=25)
# 
# # Training Assessment
# confusionMatrix(predict(fit.cheat, newdata=df.train), df.train$classe)
# 
# # Test Assessment
# confusionMatrix(predict(fit.cheat, newdata=df.test), df.test$classe)
# 
# # Predict
# predict(fit.cheat, newdata=df.pred)
# 
# # Comparison to Original Model
# compare <- data_frame(Cheating = predict(fit.cheat, newdata=df.pred),
#                       CART = predict(fit.rpart, newdata=df.pred),
#                       GBM = predict(fit.gbm, newdata=df.pred),
#                       AgreeCART = Cheating==CART, 
#                       AgreeGBM = Cheating==GBM)
# 
# compare
# compare %>% summarise(AgreementCART=mean(AgreeCART),
#                       AgreementGBM=mean(AgreeGBM))


save(list = c("df.train", "df.test", "df.pred",
              "fit.rpart", "fit.gbm"),
     file=paste(dir, "Assignments/Machine Learning/Models.RData", sep=""))