################################################################################################################
##                                                                                                            ## 
## Module     :  CS6405                                                                                       ##
## Project    :  Data Mining                                                                                  ##
## Data       :  Churn Data                                                                                   ##
## Objectives :  1) Compare different approaches to solve the supervised learning problem using the given     ## 
##                  data set.                                                                                 ##
##               2) Select the approach you believe will do best on unseen data.                              ##
##               3) Make predictions on the supplied test-set.                                                ##
## Goal       :  Accurately and reliably predict the probability that a customer churns (class = 1) for       ##
##               unseen customers.                                                                            ##
## Authors    : 1) Vikrant Siwal - 118220030                                                                  ##
##              2) Mohammad Azeem Mohammad Rafique Edrisi - 118220338                                         ##
##              3) Ankit Talwar - 118220956                                                                   ##
##              4) Sherif Baruwa - 118220341                                                                  ##
##              5) Anurag Kumar Sinha - 118220658                                                             ##
##                                                                                                            ##
################################################################################################################


## Install Required Packages.
install.packages("caret")
install.packages("pROC")
install.packages("corrplot")
install.packages("MASS")

## Declare packages to make their functions accessible in R workbook.
library(caret)
library(pROC)
library(corrplot)
library(MASS)

## Load data 'churn-train.csv' from working directory. 
churn.data <- read.csv("churn-train.csv",header=TRUE)

##------------------------------------------------------------------------------------------------------------##
## Data Exploration & Data Preparation.                                                                       ##
## The following actions will be performed:                                                                   ## 
##        - Check for missing values and clean the data before modeling                                       ##
##        - Basic analysis to understand the spread, dimensions & volume of the data                          ## 
##        - Various exploratory data analysis tools like pie chart and box plots are used to                  ##
##          identify the patterns & relationship between input variables.                                     ##
##        - Variables that impact churn rate are identified                                                   ##
##        - Check for correlation                                                                             ##
##------------------------------------------------------------------------------------------------------------## 

## Make a copy of the dataset to avoid changes made during exploratory analysis affect main dataset. 
churn.dataset <- churn.data

## Columns = 21 (Target Variable = class) , Observations = 4000
dim(churn.dataset)

## Check for missing values in the dataset.
anyNA(churn.data)
summary(is.na(churn.data))      # none

## Pie Chart plot to show the proportion percentage of customers churned & not churned
tab <- as.data.frame(table(churn.data$class))
slices <- c(tab[1,2], tab[2,2]) 
lbls <- c("Not-Churned", "Churned")
pct <- round(slices/sum(slices)*100,digits = 2)  # calculating % rounded to 2 digits
lbls <- paste(lbls, pct)                         # add percents to labels 
lbls <- paste(lbls,"%",sep="")                   # add % to labels 
pie(slices,labels = lbls, col=rainbow(length(lbls)),angle = 90, main="Percentage of Customer Churned")

##------------------------------------------------------------------------------------------------------------##
##  Pre - Processing                                                                                          ##                                                             
##------------------------------------------------------------------------------------------------------------##                                                                                                       

## There are 15 numerical variables, 4 categorical variables,1 unqiue identifier(phone number) and 1 target 
## variable. R has not identified the categorical variables, so the variables are changed manually to 
## categorical.
str(churn.dataset)
churn.dataset$state <- as.factor(churn.dataset$state)
churn.dataset$area_code <- as.factor(churn.dataset$area_code)
churn.dataset$international_plan <- as.factor(churn.dataset$international_plan)
churn.dataset$voice_mail_plan <- as.factor(churn.dataset$voice_mail_plan)
churn.dataset$class <- as.factor(churn.dataset$class)

## 'phone_number' variable is not needed in the model because its not an explanatory variable, 
## so it's dropped from the dataset and used as a row name.
drop_column <- which(colnames(churn.dataset) == "phone_number")
churn.dataset <- churn.dataset[,-drop_column]
rownames(churn.dataset) <- churn.data$phone_number

## Highly correlated numerical variables are identified using a plot of the correlation matrix.
numeric.var <- sapply(churn.dataset, is.numeric)
corr.matrix <- cor(churn.dataset[,numeric.var])
corrplot(corr.matrix, main="Correlation Plot for Numerical Variables", method="number", type="upper")
diag(corr.matrix) <- 0
highly_correlated <- findCorrelation(corr.matrix, cutoff=0.95, names=TRUE)
corr.matrix[,highly_correlated]

## From the correlation matrix, 4 pairs are found highly correlated hence dropping one of the
## correlated variables from each pairs.
## 1-total_day_charge vs total_day_minutes
## 2-total_eve_charge vs total_eve_minutes
## 3-total_night_charge vs total_night_minutes
## 4-total_intl_charge vs total_intl_minutes
churn.dataset$total_day_charge <- NULL
churn.dataset$total_eve_charge <- NULL
churn.dataset$total_night_charge <- NULL
churn.dataset$total_intl_charge <- NULL

## A boxplot is used to show and understand the relationship between number of voicemail messages
## and voice mail plan
boxplot(number_vmail_messages~voice_mail_plan,data=churn.dataset,
        main="Dependence of number_vmail_messages and voice_mail_plan", 
        xlab="Class", names=c("0 (Not Churn)","1 (Churn)"))

## number_vmail_messages is set to NULL (dropped) because of it's relationship with voice_mail_plan
churn.dataset$number_vmail_messages<-NULL

##------------------------------------------------------------------------------------------------------------##
## Feature Selection (forward and backward selection) using minimum BIC as the criteria                       ##
## for selection.                                                                                             ##
##------------------------------------------------------------------------------------------------------------##

## Fit the Null model
null.model <- glm(class ~ 1, data=churn.dataset, family=binomial, maxit=100)

## Fit the Saturated model
full.model <- glm(class ~ ., data=churn.dataset, family=binomial)

## StepAIC function is used for forward and backward selection
forward.model <-stepAIC(null.model,direction="forward",scope=list(upper=full.model,lower=null.model),
                       trace=0,k=log(nobs(full.model)))
backward.model <- stepAIC(full.model, direction='backward', trace=0,
                         k=log(nobs(full.model)))

## List of important variables from forward selection.
imp_var_forward <- attr(terms(forward.model), "term.labels")

## List of important variables from backward selection.
imp_var_backward <- attr(terms(backward.model), "term.labels")

## List of variables in both models are same, so no further analysis is needed.
sort(imp_var_forward) == sort(imp_var_backward)

## List of important variables from forward model.
imp_variables = attr(terms(forward.model), "term.labels")

## Filter the important variables that will be used to fit the model
churn.dataset <- churn.dataset[,imp_variables]
churn.dataset["class"] <- as.factor(churn.data$class)

##-------------------------------------------------------------------------------------------------------------##
## The data is partitioned in two parts. One part for training which has 70% of the data and the other part    ##
## for validation which has 30% of the data. The training data is used to train the model while the            ##
## validation data is used to test and estimate model performance.                                             ##
##-------------------------------------------------------------------------------------------------------------##

## seed is set to 9 to reproduce a particular sequence of random numbers. 
set.seed(9)
index <- createDataPartition(churn.dataset$class,p=0.7,list=FALSE)
training <- churn.dataset[index,]
validation <- churn.dataset[-index,]

## Check the proportion of each data sets
summary(training$class) #Nochurn-86.32 Churn-13.67%
summary(validation$class) #Nochurn-86.40 Churn-13.59%

##------------------------------------------------------------------------------------------------------------##
## Selection, Building, Training, Testing & Fine tuning/Improving performance of different ML models          ##
## We will seggregate this part into majorly three further parts for each Selected Model/Algorithm            ##
##   -> Build/Train Model on train dataset using "Caret" package. We will "Scale" the data using              ##
##      caret inbuild function 'preProcess' while fitting the model.                                          ##
##   -> Improve further model performance by fine tuning various model parameters.                            ##
##   -> Test the Model on validation dataset, Calculating various Model performances.                         ##    
##------------------------------------------------------------------------------------------------------------##


##------------------------------------------------------------------------------------------------------------##
## 1st Model -- Support Vector Machine (SVM)                                                                  ##
##                                                                                                            ##
## Building/Training Model on train dataset. We will use 10-fold cross validation with three repeats while    ##
## training the model. Our target variable in this model is class and the variables are:-                     ##
## international_plan, number_customer_service_calls, total_day_minutes, voice_mail_plan,total_eve_minutes,   ##
## total_intl_minutes, total_intl_calls, total_night_minutes. The selection of these variables were on the    ##
## basis of above exploratory data analysis findings.                                                         ##
##------------------------------------------------------------------------------------------------------------##

## Give names to the levels of target variable for building SVM model
levels(training$class) <- make.names(c("NoChurn","Churn"))
levels(validation$class) <- make.names(c("NoChurn","Churn"))

## We are fitting model with Linear, Radial and Polynomial kernel of SVM and will choose best kernel based on  
## its performance (Kappa metric).

## Linear SVM Model on training data. We scale the data while fitting the model.
set.seed(9)
trctrl <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
training.svm.linear <- train(class~., data = training, method = "svmLinear",
                           trControl=trctrl,
                           preProcess = c("center", "scale"),
                           metric="Kappa")

## Radial SVM Model on training data. We scale the data while fitting the model.
set.seed(9)
trctrl <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
training.svm.radial <- train(class~., data = training, method = "svmRadial",
                             trControl=trctrl,
                             preProcess = c("center", "scale"),
                             metric="Kappa")

## Polynomial SVM Model on training data. We scale the data while fitting the model.
set.seed(9)
trctrl <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
training.svm.polynomial <- train(class~., data = training, method = "svmPoly",
                             trControl=trctrl,
                             preProcess = c("center", "scale"),
                             metric="Kappa")

## Check for performance of all 3 SVM kernels using Kappa metric.
linear_svm <- training.svm.linear$results$Kappa
radial_svm <- training.svm.radial$results$Kappa[which.max(training.svm.radial$results$Kappa)]
polynomial_svm <- training.svm.polynomial$results$Kappa[which.max(training.svm.polynomial$results$Kappa)]

## Plot all 3 SVM kernel's performance (kappa).
plot(c(linear_svm,radial_svm,polynomial_svm),xaxt='n',t='b',main = "SVM Kernels Performance",
     xlab="Kernel Type", ylab="Kappa Score")
axis(1, at=1:3, labels=c("Linear","Radial","Polynomial"))
## We are using Kappa metric to choose best SVM kernel and since the polynomial kernel performs better 
## on the training data, we will use this kernel to further the SVM modelling.

## Based on the baseline SVM Polynomial model performance, we select range of SVM tuning 
## parameters using grid to tune Cost(C), Scale and Degree parameters of SVM.
trctrl <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
grid_poly <- expand.grid(C=seq(0.4,1.5,by=0.20),scale=seq(0.08,0.15,by=0.01),degree=c(2,3,4,5))
set.seed(9)
training.svm.poly.tune <- train(class~., data = training, method = "svmPoly",
                                trControl=trctrl,
                                preProcess = c("center", "scale"),
                                tuneGrid = grid_poly,
                                metric="Kappa")

## A plot to visualise the SVM polynomial parameters using Kappa metric. 
plot(training.svm.poly.tune,ylab="Kappa Score",main="SVM - Poly tuning parameters")
## Model performance is optimal with these parameters
## C = 1.2, scale = 0.1, degree = 4 so we use them of our SVM model.

## We fit SVM model with parameters that we got after tuning. We are using 10-fold Cross validation 
## with three repeats while training the model. 
trctrl <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
grid_best_poly <- expand.grid(C=1.2,scale=0.1,degree=4)
set.seed(9)
training.svm.poly.best <- train(class~., data = training, method = "svmPoly",
                                trControl = trctrl,
                                preProcess = c("center", "scale"),
                                tuneGrid = grid_best_poly,
                                metric="Kappa")

## Test SVM performance on validation data.
pred.svm.validationData <- predict(training.svm.poly.best, newdata = validation)

## Fit the SVM model with parameters that we got after tuning using class probabilities as TRUE
## to calculate ROC
trctrl.roc <- trainControl(method = "repeatedcv", number = 10,repeats = 3,classProbs = TRUE)
grid_best_poly <- expand.grid(C=1.2,scale=0.1,degree=4)
set.seed(9)
training.svm.poly.best.roc <- train(class~., data = training, method = "svmPoly",
                                    trControl=trctrl.roc,
                                    preProcess = c("center", "scale"),
                                    tuneGrid = grid_best_poly,
                                    metric="Kappa")

## Test SVM performance on validation data with class probabilities as TRUE
pred.svm.validationData.roc <- predict(training.svm.poly.best.roc, newdata = validation,type="prob")


## We create a confusion matrix to check performance of SVM model on validation data. 
## Kappa, Sensitivity, F1 score, ROC/AUC will be used as performance metrics for model selection.
confusionMatrix(pred.svm.validationData,  positive = "Churn", validation$class, mode="everything")

##------------------------------------------------------------------------------------------------------------##
## 2nd Model -- Random Forrest (RF)                                                                           ##
##                                                                                                            ##
## Build/Train Model on train dataset. We will use 10-fold Cross validation with three repeats while          ##
## training the model to limit and reduce overfitting.                                                        ##
##------------------------------------------------------------------------------------------------------------##


##------------------------------------------------------------------------------------------------------------##
## Baseline Random Forrest Model (RF)                                                                         ##
##                                                                                                            ##
## We check performance of RF model with baseline parameters for mtry and ntree. We need to keep balance      ##
## between bias and variance. We will not keep ntree baseline to be very small nor too large as it can result ##
## to be either biased or overfitted. We will check different values of mtry (1 to 8) and ntree parameter of  ##
## Random Forest using grid search and with optimal value based on Kappa Score. We will use 10-fold cross     ##
## validation with three repeats while training the model.                                                    ##
##------------------------------------------------------------------------------------------------------------##

set.seed(9)
trctrl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(1:8))
training.rf.baseline <- train(class~., data=training, method="rf", metric="Kappa",
                              preProcess = c("center", "scale"),
                              tuneGrid=tunegrid, trControl=trctrl,importance = TRUE)

## Visualise all mtry values Vs Kappa Score using a plot. 
plot(training.rf.baseline,xlab="Values of mtry",ylab="Kappa Score")

## Based on optimal mtry value which we got in the RF model, we will further check for optimal 
## values of ntree. Considering no. of observations in training model, we will check for ntree
## values between 500 to 2500 at a period of 500.
trctrl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=4) # Best mtry so far
modellist <- list()
kappa_score <- numeric(5)
ntree_values <- c(500,1000, 1500, 2000, 2500)
for (i in 1:5) {
  fit <- train(class~., data=training, method="rf", metric="Kappa", tuneGrid=tunegrid, trControl=trctrl, 
               ntree=ntree_values[i])
  key <- toString(ntree_values[i])
  modellist[[key]] <- fit
  kappa_score[i] <- fit$results$Kappa
}

## We compare Random forest's ntree tuning parameter.
## Visualise all ntree values Vs Kappa Score
plot(c(500,1000, 1500, 2000, 2500),kappa_score,t='b',xlab="ntree",ylab="Kappa Score")

## After finding the range of mtry and ntree values, we are checking for best possible combination of 
## both tuning parameters of random forest model.
trctrl <- trainControl(method="repeatedcv", number=10, repeats=3)  # train control setup
mtry <- c(1:8)
result_kappa <- matrix(NA,8,5)
result_accuracy <- matrix(NA,8,5)
ntree <- c(500,1000, 1500, 2000, 2500)

## Note this will take at least one hour to compute 
for (m in 1:8){
  tunegrid <- expand.grid(.mtry=m)
  for (i in 1:5) {
    fit <- train(class~., data=training, method="rf", metric="Kappa", tuneGrid=tunegrid, 
                 trControl=trctrl, ntree=ntree[i])
    result_kappa[m,i] <- fit$results$Kappa
    result_accuracy[m,i] <- fit$results$Accuracy
  }
}

## Change the row and column names of data frame 'result_kappa'
rownames(result_kappa)=c(1:8)
colnames(result_kappa)=c(500,1000, 1500, 2000, 2500)

## Change the row and column names of data frame 'result_accuracy'
rownames(result_accuracy)=c(1:8)
colnames(result_accuracy)=c(500,1000, 1500, 2000, 2500)

## Plot kappa score of all tried mtry and ntree values.
par(mfrow=c(1,2))
plot(result_kappa[,1],t='b',xlab="mtry",ylab="Kappa Score",col="red",pch=19,cex=2)
points(result_kappa[,2],t='b',col="green",pch=19,cex=2)
points(result_kappa[,3],t='b',col="blue",pch=19,cex=2)
points(result_kappa[,4],t='b',col="purple",pch=19,cex=2)
points(result_kappa[,5],t='b',col="brown",pch=19,cex=2)
legend(5,0.4,legend=c("500", "1000","1500","2000","2500"),
       col=c("red","green", "blue","purple","brown"), lty=1,cex=1,title='ntree')

## Plot Accuracy score of all tried mtry and ntree values.
plot(result_accuracy[,1],t='b',xlab="mtry",ylab="Accuracy Score",col="red",pch=19,cex=2)
points(result_accuracy[,2],t='b',col="green",pch=19,cex=2)
points(result_accuracy[,3],t='b',col="blue",pch=19,cex=2)
points(result_accuracy[,4],t='b',col="purple",pch=19,cex=2)
points(result_accuracy[,5],t='b',col="brown",pch=19,cex=2)
legend(5,0.91,legend=c("500", "1000","1500","2000","2500"),
       col=c("red","green", "blue","purple","brown"), lty=1,cex=1,title='ntree')

## We choose the best mtry and ntree values of random forrest
## From the graph plots, we pick best RF model which has high a kappa score and proceed with corresponding 
## mtry and ntree value.

## We fit the best RF model with parameters that we got after tuning: mtry = 3 and ntree = 500. We are 
## using 10-fold Cross validation with three repeats while training the model. We will scale the data  
## using caret inbuilt function 'preProcess' and will choose 'Kappa' metric for selecting best performance.
trctrl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=3)
training.rf.best <- train(class~., data=training, method="rf", metric="Kappa",
                          preProcess = c("center", "scale"),
                      tuneGrid=tunegrid,ntree=500, trControl=trctrl,importance = TRUE)

## Test Random Forrest performance on validation data.
pred.rf.validationData <- predict(training.rf.best, newdata = validation)

## To calculate ROC, we are fitting best Random Forrest model with parameters that we got after 
## tuning with class probabilities as TRUE.
trctrl.roc <- trainControl(method = "repeatedcv", number = 10,repeats = 3,classProbs = TRUE)
grid_best_rf <- expand.grid(.mtry=3)
set.seed(9)
training.rf.best.roc <- train(class~., data=training, method="rf", metric="Kappa",
                              preProcess = c("center", "scale"),
                              tuneGrid=grid_best_rf,ntree=500, trControl=trctrl.roc,importance = TRUE)

## Test Random Forrest performance on validation data with class probabilities as TRUE to plot ROC
pred.rf.validationData.roc <- predict(training.rf.best.roc, newdata = validation,type="prob")


## We create a confusion matrix to check performance of RF model on validation data. We will use Kappa,
## Sensitivity, F1, ROC AUC as metrics for model selection.
confusionMatrix(pred.rf.validationData,  positive = "Churn", validation$class, mode="everything")

## Compare fitted Random Forrest and SVM model performance on training data using Kappa metric at 
## 95% Confidence Level.
fitted_model_compare <- resamples(list(RF=training.rf.best, SVM=training.svm.poly.best))
summary(fitted_model_compare)
## A plot to visualize the comparison. 
dotplot(fitted_model_compare)
## Random Forrest performs better on validation data as compared to SVM model on all comparison criterias.

##------------------------------------------------------------------------------------------------------------##
## ROC-AUC of SVM plot and calculation for SVM and Random Forest model.                                       ##
##------------------------------------------------------------------------------------------------------------##
par(pty = "s")

## Calculate ROC for SVM model on validation data
svm.roc<-roc(validation$class,pred.svm.validationData.roc[,2],
             plot = TRUE,legacy.axes=TRUE, percent = TRUE, xlab="False Positive Percentage",
             ylab="True Positive Percentage",col="blue",print.auc= TRUE,lwd=4, main="ROC Curve Of Validation Data")

## Calculate ROC for Random Forrest model on validation data
rf.roc<-roc(validation$class,pred.rf.validationData.roc[,2], add = TRUE, print.auc.y=40,
            plot = TRUE,legacy.axes=TRUE, percent = TRUE, col="green", print.auc= TRUE,lwd=4)
legend("bottomright",legend=c("SVM","Random Forrest"),col=c("blue","green"),lwd=4)

par(pty = "m")

## Choosing Random Forest as the best model between the two and using same to make prediction on holdout dataset.

##------------------------------------------------------------------------------------------------------------##
##                   Prediction On Holdout Dataset                                                            ##
##------------------------------------------------------------------------------------------------------------##

## Load 'churn-holdout.csv' from our working directory.                                                       ##
##------------------------------------------------------------------------------------------------------------##
churn.holdout.data <- read.csv("churn-holdout.csv",header=TRUE)

## Make a copy of the dataset.
churn.holdout.dataset <- churn.holdout.data

## 15 Numerical, 4 categorical variables and 1 unqiue identifier(phone number).
dim(churn.holdout.dataset)          ## Columns = 20 , Observations = 1000

##------------------------------------------------------------------------------------------------------------##
##  Pre - Processing                                                                                          ##                                                             
##------------------------------------------------------------------------------------------------------------##                                                                                                       

str(churn.holdout.dataset)
## Same pre-processing done on main data is done on holdout dataset. 
## R has not identified the categorical variables, so we manually change the variables into factors indicating 
## they are categorical variables.
churn.holdout.dataset$state <- as.factor(churn.holdout.dataset$state)
churn.holdout.dataset$area_code <- as.factor(churn.holdout.dataset$area_code)
churn.holdout.dataset$international_plan <- as.factor(churn.holdout.dataset$international_plan)
churn.holdout.dataset$voice_mail_plan <- as.factor(churn.holdout.dataset$voice_mail_plan)

## 'phone_number' variable is not needed in the model because its not an explanatory variable, 
## so it's dropped from the dataset and used as a row name.
phone_number<-churn.holdout.data$phone_number                 # store the phone number
rownames(churn.holdout.dataset) <- churn.holdout.data$phone_number    # make it a row name
drop_column <- which(colnames(churn.holdout.dataset) == "phone_number")
churn.holdout.dataset <- churn.holdout.dataset[,-drop_column]

str(churn.holdout.dataset)

## Drop variables which were dropped in exploratory data analysis in training data.
churn.holdout.dataset$total_day_charge <- NULL
churn.holdout.dataset$total_eve_charge <- NULL
churn.holdout.dataset$total_night_charge <- NULL
churn.holdout.dataset$total_intl_charge <- NULL
churn.holdout.dataset$number_vmail_messages <- NULL

## Same important variables that were decided in exploratory data analysis on training data are used
churn.holdout.dataset <- churn.holdout.dataset[,imp_variables]jmb

## As Random Forrest performs better than Support Vector Machine on Validation data, random forest is used 
## to predict the holdout data set.

## Prediction on holdout data.
prediction.holdout.data <- predict(training.rf.best, newdata = churn.holdout.dataset)

## Change target Class name to desired form.
pred <- ifelse(prediction.holdout.data == "NoChurn", 0, 1)

## Predict individual class probalities of target class on holdout data.
prediction.probabilities <- predict(training.rf.best, newdata = churn.holdout.dataset, type="prob")
output <- data.frame(phone_number,pred)
output["0"] <- prediction.probabilities[,1]
output["1"] <- prediction.probabilities[,2]

## Save the final output in csv
write.csv(output, file = "predictions-118220030_118220338_118220956_118220341_118220658.csv", row.names = F)
