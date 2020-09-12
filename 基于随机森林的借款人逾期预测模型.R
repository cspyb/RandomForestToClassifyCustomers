# libraries
library(Amelia) 
library(gbm)
library(kernlab)
library(caret)
library(pROC)
library(randomForest)
# load data set
credit_data <- read.csv('C:\\Users\\1ia0\\Desktop\\鲸鲲智数-模型实习笔试题\\鲸鲲智数-模型实习笔试_dataset.csv',stringsAsFactors = T)

# 探索性数据分析
## 缺失值分析
missmap(credit_data, main = 'missmap of features in data set')

all(credit_data$is_apply == 1) #检查是否所有人都申请了贷款

sum(credit_data$is_pass != credit_data$is_loan) #有632个申请人通过了申请但是没有贷款

##申请信息变量的缺失值与is_loan之间的关系
all(which(credit_data$disbr_dt=='') == which(credit_data$is_loan==0))
all(which(credit_data$due_dt=='') == which(credit_data$is_loan==0))
all(which(credit_data$is_settle=='') == which(credit_data$is_loan==0))
all(which(credit_data$is_ovdu=='') == which(credit_data$is_loan==0))
all(which(credit_data$is_dpd7=='') == which(credit_data$is_loan==0)) #结果表明is_loan=0导致的缺失值

##变量分布
table(credit_data$is_loan) #0:28406 1:6495
table(credit_data$is_pass) #0:27774 1:7127 pass rate:20.4%


# 数据预处理
## 筛选出贷款用户
credit_data_new <- credit_data[credit_data$is_loan==1,-c(16,17,22,23)]
## 处理NA值
for(i in c(1:47)){
  print(sum(is.na(credit_data_new[,i])))
}

credit_data_new <- credit_data_new[-which(is.na(credit_data_new[,30])),]

## 处理-99
for(i in c(26:46)){
  print(sum(credit_data_new[,i]==-99))
}

credit_data_new <- credit_data_new[,-42]

for(i in c(26:46)){
  credit_data_new[,i][credit_data_new[,i]==-99] <- mean(credit_data_new[,i][credit_data_new[,i]!=-99])
}


# 划分训练集和测试集
df <- credit_data_new[,-c(1:10)]
df <- within(df, {
  is_dpd7 <- factor(is_dpd7)
  system_version <- factor(system_version)
})
train.ind <- createDataPartition(df$is_dpd7, p = 0.7)$Resample1 #1948


# 逻辑回归模型
logi_model <- glm(is_dpd7~., family = 'binomial', data = df, subset = train.ind)
summary(logi_model)
logi_pred <- predict(logi_model, df[-train.ind,],type = 'response')
roc(is_dpd7~logi_pred,data=df[-train.ind,],plot=T) #AUC=0.6018
logi_pred[logi_pred<0.5] = 0
logi_pred[logi_pred>=0.5] = 1
sum(df[-train.ind,]$is_dpd7 == logi_pred) #准确率：863/1084=79.6%
table(df[-train.ind,]$is_dpd7, logi_pred) #混淆矩阵

# SVM
svm_model <- ksvm(is_dpd7~., data = df[train.ind,], type='C-svc', C=0.01, kernel = "vanilladot")
svm_pred <- predict(svm_model, df[-train.ind,])
sum(df[-train.ind,]$is_dpd7 == svm_pred) #准确率：862/1084=79.5%
table(df[-train.ind,]$is_dpd7, svm_pred) #混淆矩阵

##系数
colSums(svm_model@xmatrix[[1]] * svm_model@coef[[1]])
a0 <- svm_model@b


# randomForest
rf_model <- randomForest(is_dpd7~., data = df[train.ind,],importance = T)
rf_pred <- predict(rf_model, df[-train.ind,])
sum(df[-train.ind,]$is_dpd7 == rf_pred) #准确率：864/1084=79.7%
table(df[-train.ind,]$is_dpd7, rf_pred) #混淆矩阵