load("D:/UCSCx/Predictive Analytics Applications of Machine Learning/Final Project/Spam data/processed data/sms_test_ncon.rda")
load("D:/UCSCx/Predictive Analytics Applications of Machine Learning/Final Project/Spam data/processed data/sms_train_ncon.rda")
load("D:/UCSCx/Predictive Analytics Applications of Machine Learning/Final Project/Spam data/processed data/sms_test_con.rda")
load("D:/UCSCx/Predictive Analytics Applications of Machine Learning/Final Project/Spam data/processed data/sms_train_con.rda")
load("D:/UCSCx/Predictive Analytics Applications of Machine Learning/Final Project/Spam data/processed data/sms_raw_train.rda")
load("D:/UCSCx/Predictive Analytics Applications of Machine Learning/Final Project/Spam data/processed data/sms_raw_test.rda")

#---------------Naive Bayes---------------#
ctrl_nb <- trainControl(method="cv", 10) #K-fold cross-validation

set.seed(12358)
sms_model1 <- train(sms_train, sms_raw_train$type, method="nb",
                    trControl=ctrl_nb)
sms_model1

#---second model---#
set.seed(12358)
sms_model2 <- train(sms_train, sms_raw_train$type, method="nb", 
                    tuneGrid=data.frame(.fL=1, .usekernel=FALSE),
                    trControl=ctrl_nb)
sms_model2

#---Testing the predictions---#
sms_predict1 <- predict(sms_model1, sms_test)
cm1 <- confusionMatrix(sms_predict1, sms_raw_test$type, positive="spam")
cm1
#---Testing the predictions with 2.model---#
sms_predict2 <- predict(sms_model2, sms_test)
cm2 <- confusionMatrix(sms_predict2, sms_raw_test$type, positive="spam")
cm2

#------------------KNN--------------------#
# Transform dtm to matrix to data frame - df is easier to work with
knn.df_train<- as.data.frame(data.matrix(sms_train_ncon), stringsAsfactors = FALSE, na.rm = T)
knn.df_test<- as.data.frame(data.matrix(sms_test_ncon), stringsAsfactors = FALSE, na.rm = T)

# Column bind category (known classification)
#knn.df_train_cl<- cbind(knn.df_train, sms_raw_train$type)
#knn.df_test_cl<- cbind(knn.df_test, sms_raw_test$type)

cl<-factor(sms_raw_train$type)
testlable<-factor(sms_raw_test$type)
cl1<-cl[1:1392]
library(class)
knn.pred <- knn(knn.df_train, knn.df_test, cl, k=1)
#knn.df_train_1<-knn.df_train[1:1392,]
table(knn.pred, testlable)

knn.pred2 <- knn(knn.df_train, knn.df_test, cl, k=2)
table(knn.pred2, testlable)

library(psych)
#cohen.kappa(table(testlable, knn.pred))
#cohen.kappa(table(testlable, knn.pred2))
confusionMatrix(testlable, knn.pred)
confusionMatrix(testlable, knn.pred2)


#--------------Decision Tree--------------#
tree.df_train = as.data.frame(data.matrix(sms_train_ncon), stringsAsfactors = FALSE, na.rm = T)
tree.df_test = as.data.frame(data.matrix(sms_test_ncon), stringsAsfactors = FALSE, na.rm = T)
RF.df_train = as.data.frame(data.matrix(sms_train_con), stringsAsfactors = FALSE, na.rm = T)
RF.df_test = as.data.frame(data.matrix(sms_test_con), stringsAsfactors = FALSE, na.rm = T)
train_cl<-factor(sms_raw_train$type)
test_cl<-factor(sms_raw_test$type)

#----------rpart----------#
library(rpart)
CARTtree = rpart(train_cl~., data = tree.df_train)
summary(CARTtree)
plot(CARTtree)
text(CARTtree, use.n = T)
prediction_CARTtree = predict(CARTtree, newdata = tree.df_test, type = 'class')
table(test_cl, prediction_CARTtree)

install.packages("psych")
library(psych)
cohen.kappa(table(test_cl, prediction_CARTtree))
confusionMatrix(test_cl, prediction_CARTtree)
#Accuracy :  0.9362
#Kappa : 0.63 <---It's kind of not pretty good

#-----------C4.5-----------#
library(RWeka)
library(grid)
library(partykit)
library(caret)
#---unpruned tree---#
C45tree_un = J48(train_cl~. , data= tree.df_train, control= Weka_control(U=TRUE))
summary(C45tree_un)
#plot(C45tree_un)
prediction_C45tree_un = predict(C45tree_un, tree.df_test)
confusionMatrix(test_cl, prediction_C45tree_un)
#Accuracy : 0.9605
#Kappa : 0.8164

#----pruned tree---#
C45tree_pru = J48(train_cl~., data = tree.df_train, control= Weka_control(U=FALSE))
summary(C45tree_pru)
#plot(C45tree_pru)
prediction_C45tree_pru = predict(C45tree_pru, tree.df_test)
confusionMatrix(test_cl, prediction_C45tree_pru)
#Accuracy : 0.9576
#Kappa : 0.8001

#-----------C50-----------#
#The aim of boosting is to increase the reliability of the predictions by performing the
#analysis iteratively and adjusting observation weights after each iteration

#install.packages("C50")
library(C50)
C50tree = C5.0(y = train_cl, x = tree.df_train, Trials = 10)
summary(C50tree)
prediction_C50tree = predict(C50tree, tree.df_test)
confusionMatrix(test_cl, prediction_C50tree)
#Accuracy : 0.9526  
#Kappa :  0.7747


#We don't have time to do this
#------Random Forest------#
#install.packages("randomForest")
#library(randomForest)
#set.seed(1234)
#RF = randomForest(tree.df_train, train_cl, ntree = 500)
#RF
#prediction_RF = predict(RF, knn.df_test)
#confusionMatrix(test_cl, prediction_RF)


##It shows what kinds of models we can use:
names(getModelInfo())
