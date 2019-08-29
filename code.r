# Set WD, read data
getwd()
setwd("C:\\Users\\Dell pc\\Music\\m u s i c c c")
hrdata = read.csv("hr.csv",header = T, na.strings=c("",NA))
head(hrdata)

# Load libraries
library(ggplot2)
library(caret)
library(irr)
library(dplyr)
library(pvclust)
library(Amelia)
library(fastDummies)
library(caTools)
library(randomForest)

#########################################
##          Data Exploration           ##
#########################################
str(hrdata)
summary(hrdata)

# Missing Values
colSums(is.na(hrdata))

# Employee Status
ggplot(hrdata, aes(x = left)) + 
geom_histogram(bins = 2, binwidth = .5, fill = 'orange') + 
labs(title = 'Employment Status Of HR Data', x = 'Status', y = 'Count')

# Distrbution by Dept
ggplot(hrdata, aes(x = sales)) +
  geom_histogram(bins = 10,stat = 'count', binwidth = .5) +
  geom_bar(aes(fill = 'red')) + 
  labs(title = 'Employee Count By Department', x = 'Department', y = 'Count')


# Distribution of Satisfaction Levels
hist(hrdata$satisfaction_level,
     breaks = 10,
     col = 'pink', 
     main = 'Distribution Of Satisfaction Level', 
     xlab = 'Satisfaction Level', 
     ylab = 'Count')

# Projects
barData <- table(as.factor(hrdata$left), hrdata$number_project)
barplot(barData, 
        main="Employees Left Vs. Projects",
        xlab="No. of Projects", 
        col=c("blue","orange"),
        legend = rownames(barData), 
        beside=T)

# Dummy Variables Creation

dummy_cols(hrdata, select_columns = hrdata$salary, remove_first_dummy = TRUE,
           remove_most_frequent_dummy = FALSE, sort_columns = FALSE,
           ignore_na = FALSE, split = NULL)
data.class(hrdata$salary)
as.factor(hrdata$salary)

sales_dummy <- dummy(hrdata$sales, sep = '_')
salary_dummy <- dummy(hrdata$salary, sep = '_')
hr2 <- cbind(hrdata, sales_dummy, salary_dummy)
ind <- which(colnames(hr2)=="sales")
hr2 <- hr2[,-ind]
ind2 <- which(colnames(hr2)=="salary")
hr2 <- hr2[,-ind2]
View(hr2)

#########################################
##        Logistic Regression          ##
#########################################


#Split data into test and training samples
set.seed(200)
index <- sample(nrow(hrdata),0.70*nrow(hrdata),replace=F)
train <- hrdata[index,]
test <- hrdata[-index,]

#Build first model using all variables 
mod <- glm(left~.,data=train,family="binomial", control = list(maxit=50))
summary(mod)
step(mod,direction="both")

# Summary Of Model
summary(mod)

# Find the best predictors
confint(mod, level = .95) 

# Model Predictions
hr_pred <- predict(mod, test, type = 'response')

# Find significant cut-off point
# Create a ROC Curve To find Cutoff Point
# sensitivity vs. specificity
modAUC <- colAUC(hr_pred, test$left, plotROC = T)
abline(h=model.AUC, col = 'red')
text(.2,.9,cex = .8, labels = paste('Optimal Cutoff: ', round(model.AUC, 4)))

# Convert Probabilities To Class
# 1 indicates the employee left,
# 0 indicates the employee stayed
# Covert model pred probabilities into classes
predclass <- ifelse(hr_pred > .7860, 1, 0)

# Create a confusion matrix
confusionMatrix(predclass, test$left)

#########################################
##         Cluster Analysis            ##
#########################################
hr_clusters <- pvclust(hr2)
plot(hr_clusters, main = 'HR Data Cluster')

#########################################
##                                     ##
##            4. Anova Tests           ##
##                                     ##
#########################################

# ANOVA: Time Spent At Company By Sales
anova <- aov(time_spend_company ~ sales, data = hrdata)
summary(anova)
TukeyHSD(anova, conf.level = .95)
ggplot(hrdata, 
       aes(y = time_spend_company, x = sales)) + 
  geom_boxplot(outlier.color = 'red', 
               outlier.size = .5,
               fill = '#4c90ff', 
               color = '#2a5fb7') + 
  labs(title = 'ANOVA: Time Spent At Company By Dept.', 
       x = 'Employee Department', 
       y = 'Time Spent At Company')

# ANOVA: SATISFACTION LEVEL BY SALARY
anova2 <- aov(satisfaction_level ~ salary, data = hrdata)
summary(anova2)
TukeyHSD(anova2)
ggplot(hrdata, aes(y = satisfaction_level, x = salary)) + 
geom_boxplot(outlier.color = 'red', outlier.size = .5,fill = c('#ff4f7d', '#4fa1ff', '#4fff95'), color = '#333333') + 
labs(title = 'ANOVA: Satisfaction Level By Salary', x = 'Salary Level', y = 'Satisfaction Level')

#########################################
##                                     ##
##           Random Forest             ##
##                                     ##
#########################################

# Create Model
rfmod <- randomForest(as.factor(left) ~ ., train, ntree = 20)
summary(rfmod)
rfpred <- predict(rfmod, test)
summary(rfpred)

# View Confusion Matrix
confusionMatrix(rfpred, test$left)
