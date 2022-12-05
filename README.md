# Practical Machine Learning_Week4

## Synopsis

From Coursera course site

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

*Prerequisites*
*Training and testing data is assumed to be downloaded to your working directory and the packages ggplot2, caret, rpart, randomForest, kernlab, ranger, e1071 and dplyr are installed*

Training and test data is available at 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

and

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

To run the code, download the R-script or the RMD file to your computer.

```{r}
library(ggplot2)
library(dplyr)
library(caret)
library(rpart)
library(randomForest)
library(kernlab)
library(ranger)
library(e1071)
library(flextable)

set.seed(12345)
```

## Data processing
### Reading and inspecting data

```{r, warning=FALSE, message=FALSE, cache=TRUE}
testdata <- read.csv(file = "pml-testing.csv", header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
traindata <- read.csv(file = "pml-training.csv", header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

p <- ggplot(traindata, aes(x = classe)) + geom_bar() +ggtitle("Distribution af variable classe")
p

dim(traindata)
```

The data set pml-training will be used for training (traindata) and testing the models. The data set pml-testing (testdata) will be used for answering the question of the project.

There are 5 outcomes of variable classe; A, B, C, D and E. The outcome 'A' is the most frequently occurring outcome of classe and the others are more or less equally occurring.

There are 19622 observations on 160 features in the training data. There are missing data in the training dataset and some variables are not useful for modelling. The first 7 features will not be used, they are irrelevant The variable we are modelling is the variable classe. 

Upon inspecting data we find multiple rows with missing data and remove these upon reading the data.

### Cleaning and preparing  data
Removing the first 7 features
```{r, warning=FALSE, message=FALSE, cache=TRUE}

traindata<-traindata[,-c(1:7)]
testdata<-testdata[,-c(1:7)]
```

We also remove features with near zero variance.

```{r, warning=FALSE, message=FALSE, cache=TRUE}
nearzero<-nearZeroVar(traindata)
traindata<-traindata[,-nearzero]
testdata<-testdata[,-nearzero]
```

Removing features with NAs using a threshold of 90% and converting classe variable to a factor
```{r, warning=FALSE, message=FALSE, cache=TRUE}
nacol <- sapply(traindata, function(x) mean(is.na(x)))> 0.90 %>% unlist()
traindata <- traindata[,nacol == FALSE]
traindata$classe=as.factor(traindata$classe)

```


Final step before modelling is to create a partition of the training data - a subset for training models and a validation set for evaluating the models and choosing a model for the test data.

```{r, warning=FALSE, message=FALSE, cache=TRUE}
inTrain<-createDataPartition(traindata$classe,p=0.6,list=FALSE )
train<-traindata[inTrain,]
validation<-traindata[-inTrain,]
```

We will consider 3 models: Random Forest, Classification and regression Tree and Support Vector Machine.

## Modelling

Random Forest
```{r, warning=FALSE, message=FALSE, cache=TRUE}
ranfor <- randomForest(formula = classe ~ ., data = train,ntree = 60)
ranfor_pred<-predict(ranfor,validation)
cnfmat<-confusionMatrix(ranfor_pred,validation$classe)
cnfmat

```

Classification and regression Tree - via Caret package. 
```{r, warning=FALSE, message=FALSE, cache=TRUE}
trcontr<-trainControl(method="repeatedcv",number=4,repeats=5,classProbs=TRUE)
rpart<-train(classe~.,data=traindata,method='rpart',trControl=trcontr,metric='Accuracy',tuneLength=4)
rpart_pred<-predict(rpart,validation)
cnfmat<-confusionMatrix(rpart_pred,validation$classe)
cnfmat

```

Support Vector Machine
```{r, warning=FALSE, message=FALSE, cache=TRUE}
svmfit<-svm(classe~.,data=traindata,kernel="radial",cost=0.1)
svmfit_pred<-predict(svmfit,validation)
cnfmat<-confusionMatrix(svmfit_pred,validation$classe)
cnfmat

```

## Results
Accuracy is highest for the Random Forest at 0.9907 and the Classification Tree has lowest accuracy 0.5226.
The Kappa statistic is highest for the Random Forest. The Kappa statistic adjusts accuracy for the possibility a correct prediction by chance alone.

According to the Confusion Matrices and the Kappa Statistcs the Random Forest is the preffered model. It is chosen as the model to use in predicting testing data. 

Predicting test cases
```{r, warning=FALSE, message=FALSE, cache=TRUE}

colnames(testdata)[118]<-"classe"

res<-data.frame(Subject=1:nrow(testdata),Classe=predict(ranfor,testdata))

ft<-flextable(res)
ft<-theme_vanilla(ft)
ft<-set_caption(ft,"Prediction on test data")
ft


```


