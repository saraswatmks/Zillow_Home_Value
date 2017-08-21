path <- "/home/manish/Desktop/Data2017/July/Zillow/"
setwd(path)

# load data
train <- fread("train_2016_v2.csv/train_2016_v2.csv", na.strings = c(""," ",NA,"NA"))
properties <- fread("properties_2016.csv/properties_2016.csv",na.strings = c(""," ",NA,"NA"))
samplesub <- fread("sample_submission.csv",header = T)

# remove rows with all missing values from properties
properties[,na_count := rowSums(is.na(.SD))]
properties[,.N,na_count]
properties <- properties[na_count < 57]

#remove duplicate columns
dups <- colnames(properties)[which(duplicated(t(properties)))] # duplicates

## remove column with same description
cols_to_rmv <- c('calculatedbathnbr','censustractandblock','calculatedfinishedsquarefeet','finishedsquarefeet50','fireplaceflag','fullbathcnt')
properties[,(cols_to_rmv) := NULL]

# merge data
train <- properties[train, on = 'parcelid']
setnames(samplesub, "ParcelId", "parcelid")
test <- properties[samplesub[,.(parcelid)], on = 'parcelid']

rm(properties)

# create a simple model and check the score
char_cols <- colnames(train)[sapply(train, is.character)]

# remove unmatching levels from train and test
for(x in colnames(train))
{
  if (x == 'transactiondate') next
    if(class(train[[x]]) == 'character')
    {
      train_a <- unique(train[[x]])
      test_a <- unique(test[[x]])
      
      remove_train <- setdiff(train_a, test_a)
      remove_test <- setdiff(test_a, train_a)
      
      remove <- c(remove_train, remove_test)
      
      k <- function(x)
      {
        if (x %in% remove) return (NA)
        else return (x)
      }
      
      train[,eval(x) := unlist(lapply(get(x), k))]
      test[,eval(x) := unlist(lapply(get(x), k))]
      
    }
    
  }

train[,transactiondate := NULL]


# remove columns with 99% missing values
duhtr <- sapply(train, function(x) sum(is.na(x))/length(x))
dunte <- sapply(test, function(x) sum(is.na(x))/length(x))

duhtr <- duhtr[duhtr > 0.98]

train[,(names(duhtr)) := NULL]
test[,(names(duhtr)) := NULL]


# convert character to integers
features <- c('propertyzoningdesc','propertycountylandusecode','hashottuborspa')
for ( f in features)
{
  if (class(test[[f]]) == 'character')
  {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.numeric(factor(train[[f]], levels = levels))
    test[[f]] <- as.numeric(factor(test[[f]], levels = levels))
  }
}


# MODEL 1 -----------------------------------------------------------------

## try xgboost
library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(train[,-c('logerror'),with=F]), label = train$logerror, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test), missing = NA)

params <- list(
  eta = 0.05,
  max_depth = 6,
  objective = 'reg:linear',
  #eval_metric = 'rmse',
  base_score = mean(train$logerror),
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
  
)

bst <- xgb.cv(params = params
              , data = dtrain
              , nrounds = 1000
              , nfold = 5L
              , print_every_n = 10
              , early_stopping_rounds = 40
              ,metrics = "mae")


bst1 <- xgb.train(params = params, data = dtrain, nrounds = 19)
preds <- predict(bst1,dtest)
preds <- round(preds, 4)

samplesub$`201610` <- preds
samplesub$`201611` <- preds
samplesub$`201612` <- preds
samplesub$`201710` <- preds
samplesub$`201711` <- preds
samplesub$`201712` <- preds

best_score <- bst$evaluation_log$test_mae_mean[19]
sub_name <- paste("jeff/", "submission_cv_",best_score, "_", Sys.time(),".csv", sep = "")
sub_name <- gsub(":","-", sub_name)
sub_name <- gsub(" ","_", sub_name)

fwrite(samplesub, sub_name) # 0.0649124 LB | 0.068134 CV

# check variable importance
xgbimp <- xgb.importance(feature_names = colnames(dtrain), model = bst1)
xgb.ggplot.importance(importance_matrix = xgbimp)



# MODEL 2 -----------------------------------------------------------------

# Create Another model by removing outlier ------------------------------

train <- train[logerror > -0.4]
train <- train[logerror < 0.419]

## try xgboost
library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(train[,-c('logerror'),with=F]), label = train$logerror, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test), missing = NA)

params <- list(
  eta = 0.1,
  max_depth = 8,
  objective = 'reg:linear',
  #eval_metric = 'rmse',
  base_score = mean(train$logerror),
  min_child_weight = 1,
  subsample = 0.7,
  colsample_bytree = 0.7
  
)

bst <- xgb.cv(params = params
              , data = dtrain
              , nrounds = 1000
              , nfold = 5L
              , print_every_n = 10
              , early_stopping_rounds = 40
              ,metrics = "mae")


bst1 <- xgb.train(params = params, data = dtrain, nrounds = 20)
preds <- predict(bst1,dtest)
preds <- round(preds, 4)

samplesub$`201610` <- preds
samplesub$`201611` <- preds
samplesub$`201612` <- preds
samplesub$`201710` <- preds
samplesub$`201711` <- preds
samplesub$`201712` <- preds

best_score <- bst$evaluation_log$test_mae_mean[which.min(bst$evaluation_log$test_mae_mean)]
sub_name <- paste("jeff/", "submission_cv_",best_score, "_", Sys.time(),".csv", sep = "")
sub_name <- gsub(":","-", sub_name)
sub_name <- gsub(" ","_", sub_name)

fwrite(samplesub, sub_name) # 0.0647577 LB  | 0.0527734 CV



# Add Features - Without Outlier Gives Better Result ----------------------
# features taken from https://www.kaggle.com/nikunjm88/creating-additional-features


train <- train[,":="(N_LivingAreaProp = finishedsquarefeet12 / lotsizesquarefeet,
                     N_ExtraSpace = lotsizesquarefeet / finishedsquarefeet12,
                     N_ValueProp = structuretaxvaluedollarcnt / landtaxvaluedollarcnt,
                     N_location = latitude + longitude,
                     N_ValueRatio = taxvaluedollarcnt / taxamount,
                     N_TaxScore = taxvaluedollarcnt * taxamount)]

train[,N_TotalRooms := bathroomcnt + bedroomcnt]
train[,N_AvRoomSize := finishedsquarefeet12 / N_TotalRooms]
train[,N_ZipCount := .N, regionidzip]
train[,N_CityCount := .N, regionidcity]
train[,N_CountyCount := .N, regionidcounty]


test <- test[,":="(N_LivingAreaProp = finishedsquarefeet12 / lotsizesquarefeet,
                     N_ExtraSpace = lotsizesquarefeet / finishedsquarefeet12,
                     N_ValueProp = structuretaxvaluedollarcnt / landtaxvaluedollarcnt,
                     N_location = latitude + longitude,
                     N_ValueRatio = taxvaluedollarcnt / taxamount,
                     N_TaxScore = taxvaluedollarcnt * taxamount)]

test[,N_TotalRooms := bathroomcnt + bedroomcnt]
test[,N_AvRoomSize := finishedsquarefeet12 / N_TotalRooms]
test[,N_ZipCount := .N, regionidzip]
test[,N_CityCount := .N, regionidcity]
test[,N_CountyCount := .N, regionidcounty]


# Model 3 -----------------------------------------------------------------

library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(train[,-c('logerror'),with=F]), label = train$logerror, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test), missing = NA)

params <- list(
  eta = 0.1,
  max_depth = 8,
  objective = 'reg:linear',
  #eval_metric = 'rmse',
  base_score = mean(train$logerror),
  min_child_weight = 1,
  subsample = 0.7,
  colsample_bytree = 0.7
  
)

bst <- xgb.cv(params = params
              , data = dtrain
              , nrounds = 1000
              , nfold = 5L
              , print_every_n = 10
              , early_stopping_rounds = 40
              ,metrics = "mae")


bst1 <- xgb.train(params = params, data = dtrain, nrounds = 17)
preds <- predict(bst1,dtest)
preds <- round(preds, 4)

samplesub$`201610` <- preds
samplesub$`201611` <- preds
samplesub$`201612` <- preds
samplesub$`201710` <- preds
samplesub$`201711` <- preds
samplesub$`201712` <- preds

best_score <- bst$evaluation_log$test_mae_mean[which.min(bst$evaluation_log$test_mae_mean)]
sub_name <- paste("jeff/", "submission_cv_",best_score, "_", Sys.time(),".csv", sep = "")
sub_name <- gsub(":","-", sub_name)
sub_name <- gsub(" ","_", sub_name)

fwrite(samplesub, sub_name)library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(train[,-c('logerror'),with=F]), label = train$logerror, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test), missing = NA)

params <- list(
  eta = 0.1,
  max_depth = 8,
  objective = 'reg:linear',
  #eval_metric = 'rmse',
  base_score = mean(train$logerror),
  min_child_weight = 1,
  subsample = 0.7,
  colsample_bytree = 0.7
  
)

bst <- xgb.cv(params = params
              , data = dtrain
              , nrounds = 1000
              , nfold = 5L
              , print_every_n = 10
              , early_stopping_rounds = 40
              ,metrics = "mae")


bst1 <- xgb.train(params = params, data = dtrain, nrounds = 20)
preds <- predict(bst1,dtest)
preds <- round(preds, 4)

samplesub$`201610` <- preds
samplesub$`201611` <- preds
samplesub$`201612` <- preds
samplesub$`201710` <- preds
samplesub$`201711` <- preds
samplesub$`201712` <- preds

best_score <- bst$evaluation_log$test_mae_mean[which.min(bst$evaluation_log$test_mae_mean)]
sub_name <- paste("jeff/", "submission_cv_",best_score, "_", Sys.time(),".csv", sep = "")
sub_name <- gsub(":","-", sub_name)
sub_name <- gsub(" ","_", sub_name)

fwrite(samplesub, sub_name) # 0.0647577 LB  | 0.527982 CV

# check variable importance
xgbimp <- xgb.importance(feature_names = colnames(dtrain), model = bst1)
xgb.ggplot.importance(importance_matrix = xgbimp)



########### Creating More Features ############################

## check count variables, and replace values occuring once or twice with NA

data_1 <- rbind(train, test, fill=T)

cols_check <- grep(pattern = ".*id",x = colnames(train), value = T)
cols_check <- append(cols_check, "yearbuilt")
cols_check <- setdiff(cols_check, "parcelid")

# for(x in cols_check)
#   set(x = data_1, j = x, value = as.character(data_1[[x]]))

data_2 <- data_1[,cols_check,with=F]
data_2 <- data_2[sample(.N, 1e5)]
data_2[, parcelid := NULL]


## remove values whose count is less than 10
for(x in cols_check)
{
  
  if(length(unique(data_1[[x]])) < 5) next
  else {
    
    p <- data_1[,.N,eval(x)]
    p <- p[N <= 10][,get(x)]
    
    to_na <- function(x)
    {
      if (x %in% p) return (-1)
      else return(x)
    }
    
    data_1[,eval(x) := unlist(lapply(get(x), to_na))]
    
  }

}


### more features

data_1[,N_garage := garagetotalsqft / garagecarcnt]
data_1[,N_squarefeet := finishedsquarefeet12/ finishedsquarefeet15]
data_1[,N_squarefeet := NULL]

data_1[,N_meancounty := mean(logerror, na.rm = T),regionidcounty]
data_1[,N_meancity := mean(logerror, na.rm = T),regionidcity]
data_1[,N_meanneighbour := mean(logerror, na.rm = T),regionidneighborhood]

data_1[,N_sdcounty := sd(logerror, na.rm = T),regionidcounty]
data_1[,N_sdcity := sd(logerror, na.rm = T),regionidcity]
data_1[,N_sdneighbour := sd(logerror, na.rm = T),regionidneighborhood]

train <- data_1[!is.na(logerror)]
test <- data_1[is.na(logerror)]

dtrain <- xgb.DMatrix(data = as.matrix(train[,-c('logerror'),with=F]), label = train$logerror, missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test), missing = NA)

params <- list(
  eta = 0.1,
  max_depth = 8,
  objective = 'reg:linear',
  #eval_metric = 'rmse',
  base_score = mean(train$logerror),
  min_child_weight = 1,
  subsample = 0.7,
  colsample_bytree = 0.7
  
)

bst <- xgb.cv(params = params
              , data = dtrain
              , nrounds = 1000
              , nfold = 5L
              , print_every_n = 10
              , early_stopping_rounds = 40
              ,metrics = "mae")


bst1 <- xgb.train(params = params, data = dtrain, nrounds = 17)
preds <- predict(bst1,dtest)
preds <- round(preds, 4)

samplesub$`201610` <- preds
samplesub$`201611` <- preds
samplesub$`201612` <- preds
samplesub$`201710` <- preds
samplesub$`201711` <- preds
samplesub$`201712` <- preds

best_score <- bst$evaluation_log$test_mae_mean[which.min(bst$evaluation_log$test_mae_mean)]
sub_name <- paste("jeff/", "submission_cv_",best_score, "_", Sys.time(),".csv", sep = "")
sub_name <- gsub(":","-", sub_name)
sub_name <- gsub(" ","_", sub_name)

fwrite(samplesub, sub_name) # 0.0814018 LB  | 0.0527734 CV







