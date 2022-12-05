library(ggplot2)
library(dplyr)
library(Hmisc)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(kernlab)
library(ranger)
library(e1071)
library(flextable)

set.seed(12345)

testdata <- read.csv(file = "pml-testing.csv", header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
traindata <- read.csv(file = "pml-training.csv", header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

p <- ggplot(traindata, aes(x = classe)) + geom_bar() + ggtitle("Distribution af variable classe")
p

dim(traindata)

traindata<-traindata[,-c(1:7)]
testdata<-testdata[,-c(1:7)]

nearzero<-nearZeroVar(traindata)
traindata<-traindata[,-nearzero]
testdata<-testdata[,-nearzero]

nacol <- sapply(traindata, function(x) mean(is.na(x)))> 0.90 %>% unlist()
traindata <- traindata[,nacol == FALSE]
traindata$classe=as.factor(traindata$classe)

inTrain<-createDataPartition(traindata$classe,p=0.6,list=FALSE )
train<-traindata[inTrain,]
validation<-traindata[-inTrain,]

ranfor <- randomForest(formula = classe ~ ., data = traindata,ntree = 60)
ranfor_pred<-predict(ranfor,validation)
cnfmat<-confusionMatrix(ranfor_pred,validation$classe)

trcontr<-trainControl(method="repeatedcv",number=4,repeats=5,classProbs=TRUE)
rpart<-train(classe~.,data=traindata,method='rpart',trControl=trcontr,metric='Accuracy',tuneLength=4)
rpart_pred<-predict(rpart,validation)
cnfmat<-confusionMatrix(rpart_pred,validation$classe)

svmfit<-svm(classe~.,data=traindata,kernel="radial",cost=0.1)
svmfit_pred<-predict(svmfit,validation)
cnfmat<-confusionMatrix(svmfit_pred,validation$classe)

colnames(testdata)[118]<-"classe"

res<-data.frame(Subject=1:nrow(testdata),Classe=predict(ranfor,testdata))

ft<-flextable(res)
ft<-theme_vanilla(ft)
ft<-set_caption(ft,"Prediction on test data")
ft
