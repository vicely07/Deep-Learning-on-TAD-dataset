library(data.table)
training <- read.csv('https://raw.githubusercontent.com/OliShawn/KmerResearch/master/4merTable/Train/4mertable.train.txt',header = TRUE)
training <- training[,names(training) != "DNA"]
head(training)
training <- training[sample(nrow(training), nrow(training)), ] #randomizes the rows
training$Class[training$Class == "1"] <- "positive"
training$Class[training$Class == "0"] <- "negative"
training$Class <- factor(training$Class)


#Preparing testing data

testing = read.csv("https://raw.githubusercontent.com/OliShawn/KmerResearch/master/4merTable/Test/4mertable.test.txt")
#testing <- read.table(file,sep=",",header=TRUE)
testing <- testing[,names(testing) != "DNA"]
testing <- testing[sample(nrow(testing), nrow(testing)), ] #randomizes the rows
testing$Class[testing$Class == "1"] <- "positive"
testing$Class[testing$Class == "0"] <- "negative"
testing$Class <- factor(testing$Class)

suppressMessages(library(caret))
suppressMessages(library(e1071))

#CARET Random Forest
do.RF <- function(training)
{  
  set.seed(313)
  n <- dim(training)[2]
  gridRF <- expand.grid(mtry = seq(from=0,by=as.integer(n/10),to=n)[-1]) #may need to change this depend on your data size
  ctrl.crossRF <- trainControl(method = "cv",number = 10,classProbs = TRUE,savePredictions = TRUE,allowParallel=TRUE)
  rf.Fit <- train(Class ~ .,data = training,method = "rf",metric = "Accuracy",preProc = c("center", "scale"),
                  ntree = 200, tuneGrid = gridRF,trControl = ctrl.crossRF)
  rf.Fit
}

#CARET Random forest
rf.Fit <- do.RF(training)
print(rf.Fit)
#predict using tuned random forest
Pred <-  predict(rf.Fit,testing)
cm <- confusionMatrix(Pred,testing$class)
print("CM for RF:") 
print(cm)
saveRDS(rf.Fit, "RF.Rds")


#Load R libraries for model generation

suppressMessages(library(caret))
suppressMessages(library(e1071))

#This is an example of CARET boosted trees using C50.
do.Boost <- function(training)
{ 
  #trials = number of boosting iterations, or (simply number of trees)
  #winnow = remove unimportant predictors
  gridBoost <- expand.grid(model="tree",trials=seq(from=1,by=2,to=100),winnow=FALSE)
  set.seed(1)
  ctrl.crossBoost <- trainControl(method = "cv",number = 10,classProbs = TRUE,savePredictions = TRUE,allowParallel=TRUE)
  C5.0.Fit <- train(Class ~ .,data = training,method = "C5.0",metric = "Accuracy",preProc = c("center", "scale"),
                    tuneGrid = gridBoost,trControl = ctrl.crossBoost)
  
  C5.0.Fit
}


#CARET boosted trees
boost.Fit <- do.Boost(training)
print(boost.Fit)
Pred <-  predict(boost.Fit,testing)
cm <- confusionMatrix(Pred,testing$class)
print("CM for Boosted:")
print(cm)
saveRDS(boost.Fit, "Boost.rds")

#CARET KNN:
grid = expand.grid(kmax=c(1:20),distance=2,kernel="optimal")
ctrl.cross <- trainControl(method="cv",number=10, classProbs=TRUE,savePredictions=TRUE)

#Requires package 'kknn' to run
knnFit.cross <- train(Class ~ .,
data = training, # training data
method ="kknn",  # model  
metric="Accuracy", #evaluation metric
preProc=c("center","scale"), # data to be scaled
tuneGrid = grid, # range of parameters to be tuned
trControl=ctrl.cross) # training controls
#print(knnFit.cross)
#plot(knnFit.cross)

#Fifth, Perform predictions on the testing set, and confusion matrix. Accuracies on testing and training should be similar.

Pred <- predict(knnFit.cross,testing)
cm<- confusionMatrix(Pred,testing$Class)
print("CM for KNN:")
print(cm)
saveRDS(knnFit.cross, "KNN.rds")

#CARET Decision Tree:
#this is based on CARET, but sometimes doesn't run well, use the e1071 instead
do.DT <- function(training)
{
  set.seed(1)
  grid <- expand.grid(cp = 2^seq(from = -30 , to= 0, by = 2) )
  ctrl.cross <- trainControl(method = "cv", number = 5,classProbs = TRUE)
  dec_tree <-   train(Class ~ ., data= Data,perProc = c("center", "scale"),
                      method = 'rpart', #rpart for classif. dec tree
                      metric ='Accuracy',
                      tuneGrid= grid, trControl = ctrl.cross
  )
  dec_tree
}
Pred <- predict(do.DT,testing)
cm<- confusionMatrix(Pred,testing$Class)
print("CM for DT:")
print(cm)
saveRDS(do.DT, "DT.rds")