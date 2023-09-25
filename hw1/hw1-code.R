#### Q1 ####
##### dependencies and env. settings #####
library(reticulate)
use_condaenv('jiaxin')

library(dplyr)
library(tidyr)
library(ggplot2)
library(GGally)
library(e1071)
library(caret)

library(parallel)
numCores = detectCores()

library(doParallel)
registerDoParallel(numCores-1)

library(keras)
library(tensorflow)
tf <- import('tensorflow')
seed = 0590
set.seed(seed)
py_set_seed(seed, disable_hash_randomization = TRUE)

set.a.seed = function(){
  set_random_seed( 
    seed, 
    disable_gpu = TRUE 
  ) 
}

##### preprocessing #####
data = read.csv('sdss-all-c.csv', header = F)
colname = c('id','class','brightness','size','texture','me1','me2')
colnames(data) <- colname
data$class[data$class == 3] <- 0 # galaxy
data$class[data$class == 6] <- 1 # star
data$me1 = as.numeric(data$me1)
data$me2 = as.numeric(data$me2)

## drop NA observations
data_complete7 = data %>% drop_na()
data_complete5 = data[,1:5]

## standardized
data_complete7[,3:7] = scale(data_complete7[,3:7])
data_complete5[,3:5] = scale(data_complete5[,3:5])

## pack datasets
packing = function(data){
  
  set.a.seed()
  ## randomly partition the dataset (.75 training)
  num = dim(data)[1]
  training_idx = sample(1:num, num*0.75, replace = F)
  data.train <-  data[training_idx,]
  data.test <-  data[-training_idx,]
  
  ## gathering data
  train.inputs <- data.train[,3:dim(data.train)[2]]
  train.targets = as.matrix(data.train$class)
  test.inputs <- data.test[, 3:dim(data.test)[2]]
  test.targets = as.matrix(data.test$class)
  
  dataset = list()
  dataset$train = list(input = train.inputs, targets = train.targets)
  dataset$test = list(input = test.inputs, targets = test.targets)
  dataset$trainsvm = data.train
  dataset$testvsm = data.test
  
  return(dataset)
}

## check classification
table(data_complete7$class)

## get dataset
dataset = packing(data_complete7)
str(dataset)

##### linear classification by tensorflow #####
### wrapped functions
sq_loss = function(targets, pred){
  tf$reduce_mean(tf$square(tf$subtract(targets, pred)))
}

# model
model = function(inputs){tf$matmul(inputs, w) + b}

## run step
run = function(t_input, targets, lr){
  with(tf$GradientTape() %as% tape, {
    pred <- model(t_input)
    loss <- sq_loss(targets, pred)
  })
  update <- tape$gradient(loss, list(w=w, b=b))
  
  w$assign_sub(update$w * lr)
  b$assign_sub(update$b * lr)
  loss
}

## training
model_train = function(t_input, targets, lr, printwidth=5){
  step = 0
  thres = 1e-6
  curr = 0
  
  repeat{
    
    step <- step + 1
    t_loss <- run(t_input, targets, lr)
    loss <- as.numeric(sprintf("%.3f\n", t_loss))
    
    if (step %% printwidth == 0) cat(sprintf("Loss at step %s: %.6f\n", step, t_loss))
    updates <- abs(loss - curr)
    
    if (step > 0 & updates <= thres) {
      cat(sprintf('Finish after %s steps.\n', step))
      break
    }
    curr <- loss
  }
}

## evaluation
eval = function(targets, pred){
  mse = sum(targets - pred)^2 / dim(targets)[1]
  pred = ifelse(pred >= 0.5, 1, 0)
  acc = sum(targets == pred)
  cat(sprintf('Accuracy: %.6f(%s/%s)  MSE: %.6f\n\n', acc/dim(targets)[1],acc, dim(targets)[1], mse))
  targets = as.factor(targets)
  pred = as.factor(round(pred))
  print(confusionMatrix(pred, targets)$table)
}

### training result (Qa.i)
lr_list = c(0.01, 0.05, 0.1, 0.15, 0.2)

input_dim = dim(dataset$train$input)[2]
output_dim = dim(dataset$train$targets)[2]

for (lr in lr_list){
  cat('Learning rate: ', lr,'\n')
  
  # initialize hidden layer parameters
  set.a.seed()
  w <-  tf$Variable(tf$random$uniform(shape(input_dim, output_dim)))
  b <-  tf$Variable(tf$zeros(shape(output_dim)))
  train.t_input = as_tensor(dataset$train$input, dtype='float32')
  
  # train
  model_train(train.t_input, dataset$train$targets, lr)
  
  # eval
  test.t_input = as_tensor(dataset$test$input, dtype='float32')
  prediction = as.vector(model(test.t_input))
  eval(dataset$test$targets, prediction)
  cat('-------------------------------------\n')
}

##### classification by svm (Qa.ii, Qa.iii) #####
formula = class~brightness+size+texture+me1+me2

set.seed(seed)
kernels = c('linear', 'radial','polynomial','sigmoid')
costs = c(0.001, 0.01, 0.1, 1,5,10,100)


# parallel tuning process
params = expand.grid(kernel = kernels, cost = costs)
results = list()
foreach(i = 1:nrow(params), .packages = 'e1071') %do% {
  current_cost = params$cost[i]
  current_kernel = params$kernel[i]
  curResult <- tune(method = svm, 
                    train.x = formula, 
                    data = dataset$trainsvm, 
                    kernel = as.character(current_kernel), 
                    cost = current_cost)
  results[[i]] <- c(current_kernel, current_cost, curResult$performances$error)
}

tuning_result = data.frame(Kernel = levels(params$kernel)[results[[i]]][1],
                           Cost = results[[i]][2], 
                           Error = results[[i]][3])

# packing results
for (i in 2:nrow(params)){
  tuning_result = rbind(tuning_result,
                        c(levels(params$kernel)[results[[i]]][1],
                          results[[i]][2], 
                          results[[i]][3]))
}

tuning_result$Kernel = as.factor(tuning_result$Kernel)
tuning_result$Cost = as.factor(tuning_result$Cost)
tuning_result$Error = as.numeric(tuning_result$Error)

tuning_result[order(tuning_result$Error, decreasing = F),]


best.svm = svm(formula, dataset$trainsvm, kernel = 'radial', cost = 100)
summary(best.svm)

pred = predict(best.svm, dataset$testvsm)
eval(dataset$testvsm$class, pred)

#### Q1- drop column me1 and me2 (Qb) ####
dataset = packing(data_complete5)

##### linear classification #####
lr_list = c(0.01, 0.05, 0.1, 0.15, 0.2)
input_dim = dim(dataset$train$input)[2]
output_dim = dim(dataset$train$targets)[2]

for (lr in lr_list){
  cat('Learning rate: ', lr,'\n')
  
  # initialize hidden layer parameters
  set.a.seed()
  w <-  tf$Variable(tf$random$uniform(shape(input_dim, output_dim)))
  b <-  tf$Variable(tf$zeros(shape(output_dim)))
  train.t_input = as_tensor(dataset$train$input, dtype='float32')
  
  # train
  model_train(train.t_input, dataset$train$targets, lr)
  
  # eval
  test.t_input = as_tensor(dataset$test$input, dtype='float32')
  prediction = as.vector(model(test.t_input))
  eval(dataset$test$targets, prediction)
  cat('-------------------------------------\n')
}


##### svm #####
formula = class~brightness+size+texture

set.seed(seed)
kernels = c('linear', 'radial','polynomial','sigmoid')
costs = c(0.001, 0.01, 0.1, 1,5,10,100)


# parallel tuning process
params = expand.grid(kernel = kernels, cost = costs)
results = list()
foreach(i = 1:nrow(params), .packages = 'e1071') %do% {
  current_cost = params$cost[i]
  current_kernel = params$kernel[i]
  curResult <- tune(method = svm, 
                    train.x = formula, 
                    data = dataset$trainsvm, 
                    kernel = as.character(current_kernel), 
                    cost = current_cost)
  results[[i]] <- c(current_kernel, current_cost, curResult$performances$error)
}

tuning_result = data.frame(Kernel = levels(params$kernel)[results[[i]]][1],
                           Cost = results[[i]][2], 
                           Error = results[[i]][3])

# packing results
for (i in 2:nrow(params)){
  tuning_result = rbind(tuning_result,
                        c(levels(params$kernel)[results[[i]]][1],
                          results[[i]][2], 
                          results[[i]][3]))
}

tuning_result$Kernel = as.factor(tuning_result$Kernel)
tuning_result$Cost = as.factor(tuning_result$Cost)
tuning_result$Error = as.numeric(tuning_result$Error)

tuning_result[order(tuning_result$Error, decreasing = F),]

best.svm = svm(formula, dataset$trainsvm, kernel = 'radial', cost = 100)
summary(best.svm)

pred = predict(best.svm, dataset$testvsm)
eval(dataset$testvsm$class, pred)


#### Q2 ####

##### preprocessing #####
x = read.table('ziptrain.dat')
y = read.table('zipdigit.dat')

## reformat input (2000 rows, 256 columns)
data <- cbind(y, x) 


## create 75 25 split by class
set.seed(seed)
split <- createDataPartition(y = data[,1], p = 0.75) |> unlist()
train <- data[split,]
test <- data[-split,]
rm(split, data)

## set up data sets, scale training data set
train <- list(input = train[,2:ncol(train)], targets = train[,1])
train[['input']] <- train[['input']] |> sapply(as.numeric) |> as.matrix()
train[['targets']] <- train[['targets']] |> as.matrix()
scale <- train[['input']] |> preProcess(method = c('center', 'scale'))
train[['input']] <- predict(scale, train[['input']])

## optionally scale test data based on training data
test <- list(input = test[,2:ncol(test)], targets = test[,1])
test[['input']] <- test[['input']] |> sapply(as.numeric) |> as.matrix()
test[['targets']] <- test[['targets']] |> as.matrix()
test[['input']] <- predict(scale, test[['input']]) |> as.matrix()

## make sure distribution of digits is similar between training and test sets
par(mfrow = c(1,2))
hist(train$targets)
hist(test$targets)

##### PCA (Qa.i)#####
# wrapped function
summary_image_by = function(func){
  summary_img = array(dim = c(10, 256))
  for (i in 0:9){
    summary_img[i+1,] <- apply(x[which(y == i),], 2, func)
  }
  summary_img
}

mean_img = summary_image_by(mean)

# remove mean effect
mean_effect = array(dim = c(nrow(x), 256))
for (i in 0:9){
  nrows = length(which(y == i))
  mean_effect[which(y == i),] <- matrix(rep(mean_img[i+1,], nrows), nrow = nrows, byrow = T)
}

cleaned_zip_img = x - mean_effect

zip.pca = prcomp(cleaned_zip_img)

curr_var = 0
Divider = TRUE
total_var = sum((zip.pca$sdev)^2)
compress_idx = 1

cat("Cumulative Importance:\n")
for (i in 1:length(zip.pca$sdev)){
  curr_var = curr_var + zip.pca$sdev[i]^2
  cat(sprintf("%s - %.2f\n", i, curr_var/total_var*100))
  if (curr_var / total_var > 0.8 & Divider){
    cat('-----------------------------------------\n')
    Divider = FALSE
    compress_idx = i
    break
  }
}


##### low rank representation #####
# wrapped function
replace_d = function(d,k){
  d_k = diag(0, nrow = length(d))
  d_k <- rep(0, length(d)) 
  d_k[1:k] <- d[1:k] 
  diag(d_k)
}

renewed_img_of = function(d_k){
  decomp_img = svd_res$u %*% d_k %*% t(svd_res$v)
  renewed_img = decomp_img + mean_effect
  dim(renewed_img) <- c(nrow(x), 16, 16)
  renewed_img
}

# svd
svd_res = svd(cleaned_zip_img)

# display
d_k = replace_d(svd_res$d, 40)
new_img = renewed_img_of(d_k)

# raw form
par(mar = rep(0.05,4), mfrow = c(40,50))
raw_img = as.matrix(x)
dim(raw_img) <- c(nrow(x), 16, 16)
apply(raw_img, 1, function(x) {image(x[,16:1], axes = F, col = gray(31:0/31))})

# low-rank form
par(mar = rep(0.05,4), mfrow = c(40,50))
apply(new_img, 1, function(x) {image(x[,16:1], axes = F, col = gray(31:0/31))})

##### linear classification #####
# evaluation function for 10 classes
eval10 = function(targets, pred){
  mse = sum(targets - pred)^2 / dim(targets)[1]
  pred = ifelse(pred >= 9, 9, ifelse(pred <= 0, 0, pred))
  pred = round(pred)
  acc = sum(targets == pred)
  cat(sprintf('Accuracy: %.6f(%s/%s)  MSE: %.6f\n\n', acc/dim(targets)[1],acc, dim(targets)[1], mse))
  targets = as.factor(targets)
  pred = as.factor(round(pred))
  print(confusionMatrix(pred, targets)$table)
}

# packing data
dataset = list()
dataset$train = train
dataset$test = test

lr_list = c(5e-4, 1e-3, 5e-3, 0.01, 0.02)

input_dim = dim(dataset$train$input)[2]
output_dim = dim(dataset$train$targets)[2]

for (lr in lr_list){
  cat('Learning rate: ', lr,'\n')
  
  # initialize hidden layer parameters
  set.a.seed()
  w <-  tf$Variable(tf$random$uniform(shape(input_dim, output_dim)))
  b <-  tf$Variable(tf$zeros(shape(output_dim)))
  train.t_input = as_tensor(dataset$train$input, dtype='float32')
  
  # train
  model_train(train.t_input, dataset$train$targets, lr, printwidth = 100)
  
  # eval
  test.t_input = as_tensor(dataset$test$input, dtype='float32')
  prediction = as.vector(model(test.t_input))
  eval10(dataset$test$targets, prediction)
  cat('-------------------------------------\n')
}

##### svm #####
colnames(dataset$train$targets) <- c('class')
colnames(dataset$test$targets) <- c('class')
dataset$trainsvm = data.frame(cbind(dataset$train$targets, dataset$train$input))
dataset$testsvm = data.frame(cbind(dataset$test$targets, dataset$test$input))

set.seed(seed)
kernels = c('linear', 'radial','polynomial','sigmoid')
costs = c(0.001, 0.01, 0.1, 1,5,10,100)


# parallel tuning process
## run each model type with generic cost = 10

results = list()
foreach(i = 1:4, .packages = 'e1071') %do% {
  current_kernel = kernels[i]
  curResult <- tune(method = svm, 
                    train.x = class~., 
                    data = dataset$trainsvm, 
                    kernel = as.character(current_kernel), 
                    cost = 10)
  results[[i]] <- c(current_kernel, 10, curResult$performances$error)
}

tuning_result = data.frame(Kernel = results[[1]][1],
                           Cost = results[[1]][2], 
                           Error = results[[1]][3])

# packing results
for (i in 2:length(kernels)){
  tuning_result = rbind(tuning_result,
                        c(results[[i]][1],
                          results[[i]][2], 
                          results[[i]][3]))
}

tuning_result$Kernel = as.factor(tuning_result$Kernel)
tuning_result$Cost = as.factor(tuning_result$Cost)
tuning_result$Error = as.numeric(tuning_result$Error)

tuning_result[order(tuning_result$Error, decreasing = F),]

## further tuning
svmfit  <- tune(method = svm, 
                train.x = dataset$train$input,
                train.y = dataset$train$targets,
                ranges = list(cost = c(0.1, 1,5,10,100),
                              kernel = c('linear', 'radial', 'polynomial', 'sigmoid')),
                scale = FALSE)


summary(svmfit$best.model)
best.svm = svm(class~., dataset$trainsvm, kernel = 'radial', cost = 100)
pred = predict(best.svm, dataset$testsvm)
eval10(dataset$test$targets, pred)