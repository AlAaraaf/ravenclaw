#### dependencies ####

library(reticulate)
use_condaenv('connectr')
library(tensorflow)
tf <- import('tensorflow')
library(keras)
tf$config$run_functions_eagerly(T)

seed = 0590
set.seed(seed)
py_set_seed(seed, disable_hash_randomization = TRUE)

set.a.seed = function(){
  set_random_seed( 
    seed, 
    disable_gpu = TRUE 
  ) 
}

library(magick)
library(PET)
library(waveslim)
library(jpeg)

library(MASS)
library(e1071)

#### data preprocessing ####
##### scale down to 16*16 and save small images #####
train_dir = './afhq/train/'
val_dir = './afhq/val/'
class = c('cat','dog','wild')

# the reduced images stored in ./dataset folder

newtrain_dir = './dataset/train/'
newval_dir = './dataset/val/'

for (current_class in class){
  current_folder = paste(val_dir, paste(current_class,'/',sep = ''),sep = '')
  filelist = list.files(current_folder)
  for (current_file in filelist){
    input_dir = paste(current_folder, current_file, sep = '')
    output_dir = paste(newval_dir, paste(current_class,'/',current_file,sep = ''),sep = '')
    img = image_read(input_dir)
    
    simg = image_scale(image_scale(img, '16'),'16')
    image_write(simg, output_dir)
  }
}

##### radon and DWT transformation #####
train_dir = './afhq/train/'
val_dir = './afhq/val/'
class = c('cat','dog','wild')

# the reduced images stored in ./dataset folder

newtrain_dir = './dataset2/train/'
newval_dir = './dataset2/val/'

for (current_class in class){
  cat('Current class: ', current_class, ' - ')
  idx = 1
  current_folder = paste(val_dir, paste(current_class,'/',sep = ''),sep = '')
  filelist = list.files(current_folder)
  for (current_file in filelist){
    input_dir = paste(current_folder, current_file, sep = '')
    output_dir = paste(newval_dir, paste(current_class,'/',current_file,sep = ''),sep = '')
    img = readJPEG(input_dir)
    simg = array(0, dim = c(20,64,3))
    for (i in 1:3){
      x.radon <- radon(oData=img[,,i], RhoSamples = nrow(img[,,i]), ThetaSamples = 160)$rData
      x.trans <- dwt.2d(x.radon, wf = "d4", J = 3)$LL3
      simg[,,i] <- x.trans
    }
    writeJPEG(simg, output_dir)
    cat(idx, ' ')
    idx = idx + 1
  }
}

#### wrapped functions ####
read.in.all.file <- function(folders, class){
  # INPUT
  # class - the class vector including all required class for classification
  
  # OUTPUT
  # filelist - the dataframe [filedir, class]
  
  filelist = data.frame()
  for (current_folder in folders){
    for (i in 0:(length(class)-1)){
      current_dir = paste(current_folder, class[i+1], '/',sep = '')
      allfiles = list.files(current_dir)
      allfiles = unlist(lapply(allfiles, function(x) {paste(current_dir, x, sep = '')}))
      current_filelist = data.frame(filedir = allfiles, 
                                    class = rep(i, length(allfiles)))
      filelist = rbind(filelist, current_filelist)
    }
  }
  
  return(filelist)
}

make.dataset <- function(n, filelist, p, img_width = 16, img_height = 16){
  # INPUT
  # n - the total number of samples (including train and test sets)
  # filelist - the dataframe [filedir, class]
  # p - the percentage of training data to total data
  
  # OUTPUT
  # the dataset list [train[data, tag], test[data, tag]]
  
  set.a.seed()
  filelist = filelist[sample(dim(filelist)[1], n),]
  train_id = sample(n, n*p)
  train_file = filelist[train_id,]
  test_file = filelist[-train_id,]
  
  dataset = list()
  trainset = list()
  trainset$data = array(0, dim = c(length(train_id), img_width, img_height, 3))
  testset = list()
  testset$data = array(0, dim = c(dim(test_file)[1], img_width, img_height, 3))
  
  for (i in 1:dim(train_file)[1]){
    current_file = train_file[i,]
    current_img = readJPEG(current_file$filedir)
    trainset$data[i,,,] <- current_img
  }
  trainset$class = train_file$class
  
  for (i in 1:dim(test_file)[1]){
    current_file = test_file[i,]
    current_img = readJPEG(current_file$filedir)
    testset$data[i,,,] <- current_img
  }
  testset$class = test_file$class
  
  dataset$train <-  trainset
  dataset$test <-  testset
  
  return(dataset)
}

convert_onehot <- function(dataset){
  dataset$train$class = to_categorical(dataset$train$class)
  dataset$test$class = to_categorical(dataset$test$class)
  
  return(dataset)
}

#### model construction ####
construct_fc_model <- function(nlayers, unitlist, 
                               img_width = 16, img_height = 16, dropout = 0.5){
  model <- keras_model_sequential()
  model %>% layer_flatten(input_shape = c(img_width,img_height,3))
  
  for (i in 1:nlayers){
    model %>% 
      layer_dense(units = unitlist[i], activation = 'relu') %>% 
      layer_batch_normalization()
  }
  
  model %>% 
    layer_dense(1, activation = 'sigmoid')
  
  return(model)
}

construct_fc_model_multiclass <- function(nlayers, unitlist,
                                          img_width = 16, img_height = 16, dropout = 0.5){
  model <- keras_model_sequential()
  model %>% layer_flatten(input_shape = c(img_width,img_height,3))
  
  for (i in 1:nlayers){
    model %>% layer_dense(units = unitlist[i], activation = 'relu')
  }
  
  model %>%
    layer_dense(3, activation = 'softmax')
  
  return(model)
}

#### binary classification ####
##### using 16*16 images #####
classlist <- c('cat','dog')
folders = c('./dataset/train/','./dataset/val/')

filelist = read.in.all.file(folders, classlist)
dataset = make.dataset(dim(filelist)[1], filelist, 0.75)

layerlist = c(2,4,8,12,16)
epochs = c(10,10,10,20,20)
unitlists =list()
unitlists[[1]] <- c(64,32) # 2 layers
unitlists[[2]] <- c(64,64,32,32) # 4 layers
unitlists[[3]] <- c(rep(64,3),rep(32,5)) # 8 layers
unitlists[[4]] <- c(rep(64,3),rep(32,6), rep(16,3)) # 12 layers
unitlists[[5]] <- c(rep(64,3),rep(32,9), rep(16,4)) # 16 layers


# model fit
checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = c(0, 0.5)

for (dp in dropout_rate){
for (lr in lrlist){
  for (i in 1:5){
    model_name = paste(checkpoint_folder, '2class_', i,'layer', '_',lr,'_', dp, sep = '')
    model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                save_weights_only = T,
                                                save_best_only = T,
                                                monitor = 'val_accuracy',
                                                mode = 'max')
    
    model <- construct_fc_model(layerlist[i], unitlists[[i]], dropout = dropout_rate)
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    historys[[i]] <- model |>
      fit(x = dataset$train$data, 
          y = dataset$train$class,
          epochs = epochs[i], batch_size = 64, validation_split = 0.2,
          callbacks = list(model_cp))
  }
}
}

# model evaluate
for (dp in dropout_rate){
for (lr in lrlist){
  cat("current lr:", lr, '\n')
  for (i in 1:5){
    model_name = paste(checkpoint_folder, '2class_', i,'layer','_',lr,'_', dp, sep = '')
    model <- construct_fc_model(layerlist[i], unitlists[[i]], dropout = dropout_rate)
    
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    model %>% load_model_weights_tf(filepath = model_name)
    model %>% evaluate(dataset$test$data, dataset$test$class, batch_size = 64)
  }
}
}

##### transformed images #####
folders2 = c('./dataset2/train/','./dataset2/val/')

filelist2 = read.in.all.file(folders2, classlist)
dataset2 = make.dataset(dim(filelist2)[1], filelist2, 0.75, 20, 64)


# hyper parameter list
layerlist = c(2,4,8,12,16)
epochs = c(10,10,10,20,20)
unitlists =list()
unitlists[[1]] <- c(32,16) # 2 layers
unitlists[[2]] <- c(32,32,16,16) # 4 layers
unitlists[[3]] <- c(rep(32,5),rep(16,3)) # 8 layers
unitlists[[4]] <- c(rep(32,3),rep(32,6), rep(16,3)) # 12 layers
unitlists[[5]] <- c(rep(32,3),rep(32,9), rep(16,4)) # 16 layers


# model fit
checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = c(0, 0.5)

for (dp in dropout_rate){
for (lr in lrlist){
  for (i in 1:5){
    model_name = paste(checkpoint_folder, '2class_', i,'layer', '_',lr,'_', dp,sep = '')
    model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                save_weights_only = T,
                                                save_best_only = T,
                                                monitor = 'val_accuracy',
                                                mode = 'max')
    
    model <- construct_fc_model(layerlist[i], unitlists[[i]], 
                                img_width = 20, img_height = 64,
                                dropout = dropout_rate)
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    historys[[i]] <- model |>
      fit(x = dataset2$train$data, 
          y = dataset2$train$class,
          epochs = epochs[i], batch_size = 64, validation_split = 0.2,
          callbacks = list(model_cp))
  }
}
}

# model evaluate
for (dp in dropout_rate){
for (lr in lrlist){
  cat("current lr:", lr, '\n')
  for (i in 1:5){
    model_name = paste(checkpoint_folder, '2class_', i,'layer','_',lr,'_', dp, sep = '')
    model <- construct_fc_model(layerlist[i], unitlists[[i]], 
                                img_width = 20, img_height = 64,
                                dropout = dropout_rate)
    
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    model %>%  load_model_weights_tf(filepath = model_name)
    model %>% evaluate(dataset2$test$data, dataset2$test$class, batch_size = 64, verbose = '2')
  }
}
}

##### flatten images for lda and svm #####
data.to.2d <- function(dataset){
  dataset$train$data <- R.utils::wrap(dataset$train$data, map = list(1, NA))
  dataset$test$data <- R.utils::wrap(dataset$test$data, map = list(1, NA))
  
  return(dataset)
}

fdataset = data.to.2d(dataset2)

##### lda #####
lda_model = lda(x = fdataset$train$data, grouping = fdataset$train$class)
lda_pred = predict(lda_model, fdataset$test$data)$class
lda_pred = as.numeric(levels(lda_pred))[lda_pred]

acc = 0
testlen = length(fdataset$test$class)
for (i in 1:testlen){
  if(lda_pred[i] == fdataset$test$class[i]){
    acc = acc + 1
  }
}
acc / testlen


##### svm #####
svm_model = svm(x = fdataset$train$data, y = as.factor(fdataset$train$class))
summary(svm_model)

svm_pred = predict(svm_model, fdataset$test$data)
svm_pred = as.numeric(levels(svm_pred))[svm_pred]

acc = 0
testlen = length(fdataset$test$class)
for (i in 1:testlen){
  if(svm_pred[i] == fdataset$test$class[i]){
    acc = acc + 1
  }
}
acc / testlen

#### multiclass classification #####
classlist <- c('cat','dog','wild')
folders = c('./dataset/train/','./dataset/val/')

filelist = read.in.all.file(folders, classlist)
dataset = make.dataset(dim(filelist)[1], filelist, 0.75)
dataset = convert_onehot(dataset)

# hyper parameter list
layerlist = c(2,4,8,12,16)
epochs = c(10,10,10,20,20)
unitlists =list()
unitlists[[1]] <- c(64,32) # 2 layers
unitlists[[2]] <- c(64,64,32,32) # 4 layers
unitlists[[3]] <- c(rep(64,3),rep(32,5)) # 8 layers
unitlists[[4]] <- c(rep(64,3),rep(32,6), rep(16,3)) # 12 layers
unitlists[[5]] <- c(rep(64,3),rep(32,9), rep(16,4)) # 16 layers


# model fit
checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = c(0,0.5)

for(dp in dropout_rate){
  for (lr in lrlist){
    for (i in 1:5){
      model_name = paste(checkpoint_folder, '3class_', i,'layer', '_',lr,'_',dp,sep = '')
      model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                  save_weights_only = T,
                                                  save_best_only = T,
                                                  monitor = 'val_accuracy',
                                                  mode = 'max')
      
      model <- construct_fc_model_multiclass(layerlist[i], unitlists[[i]], dropout = dp)
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      historys[[i]] <- model |>
        fit(x = dataset$train$data, 
            y = dataset$train$class,
            epochs = epochs[i], batch_size = 64, validation_split = 0.2,
            callbacks = list(model_cp))
    }
  }
}

# model evaluate
for (dp in dropout_rate){
  for (lr in lrlist){
    cat("current lr:", lr, 'dropout: ', dp, '\n')
    for (i in 1:5){
      model_name = paste(checkpoint_folder, '3class_', i,'layer','_',lr, '_',dp, sep = '')
      model <- construct_fc_model_multiclass(layerlist[i], unitlists[[i]], dropout = dp)
      
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      model %>% load_model_weights_tf(filepath = model_name)
      model %>% evaluate(dataset$test$data, dataset$test$class, batch_size = 64)
    }
  }
}

##### transformed images #####
folders2 = c('./dataset2/train/','./dataset2/val/')

filelist2 = read.in.all.file(folders2, classlist)
dataset2 = make.dataset(dim(filelist2)[1], filelist2, 0.75, 20, 64)
dataset2 = convert_onehot(dataset2)

# hyper parameter list
layerlist = c(2,4,8,12,16)
epochs = c(10,10,10,20,20)
unitlists =list()
unitlists[[1]] <- c(32,16) # 2 layers
unitlists[[2]] <- c(32,32,16,16) # 4 layers
unitlists[[3]] <- c(rep(32,5),rep(16,3)) # 8 layers
unitlists[[4]] <- c(rep(32,3),rep(32,6), rep(16,3)) # 12 layers
unitlists[[5]] <- c(rep(32,3),rep(32,9), rep(16,4)) # 16 layers


# model fit
checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = c(0, 0.5)

for (dp in dropout_rate){
  for (lr in lrlist){
    for (i in 1:5){
      model_name = paste(checkpoint_folder, '2class_', i,'layer', '_',lr,'_', dp,sep = '')
      model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                  save_weights_only = T,
                                                  save_best_only = T,
                                                  monitor = 'val_accuracy',
                                                  mode = 'max')
      
      model <- construct_fc_model_multiclass(layerlist[i], unitlists[[i]], 
                                  img_width = 20, img_height = 64,
                                  dropout = dropout_rate)
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      historys[[i]] <- model |>
        fit(x = dataset2$train$data, 
            y = dataset2$train$class,
            epochs = epochs[i], batch_size = 64, validation_split = 0.2,
            callbacks = list(model_cp))
    }
  }
}

# model evaluate
for (dp in dropout_rate){
  for (lr in lrlist){
    cat("current lr:", lr, '\n')
    for (i in 1:5){
      model_name = paste(checkpoint_folder, '2class_', i,'layer','_',lr,'_', dp, sep = '')
      model <- construct_fc_model_multiclass(layerlist[i], unitlists[[i]], 
                                  img_width = 20, img_height = 64,
                                  dropout = dropout_rate)
      
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      model %>%  load_model_weights_tf(filepath = model_name)
      model %>% evaluate(dataset2$test$data, dataset2$test$class, batch_size = 64, verbose = '2')
    }
  }
}



##### flatten images for lda and svm #####
fdataset2 = data.to.2d(dataset2)

##### lda #####
lda_model2 = lda(x = fdataset2$train$data, grouping = fdataset2$train$class)
lda_pred2 = predict(lda_model2, fdataset2$test$data)$class
lda_pred2 = as.numeric(levels(lda_pred2))[lda_pred2]

acc2 = 0
testlen2 = length(fdataset2$test$class)
for (i in 1:testlen2){
  if(lda_pred2[i] == fdataset2$test$class[i]){
    acc2 = acc2 + 1
  }
}
acc2 / testlen2

##### svm #####
svm_model2 = svm(x = fdataset2$train$data, y = as.factor(fdataset2$train$class))
summary(svm_model2)

svm_pred2 = predict(svm_model2, fdataset2$test$data)
svm_pred2 = as.numeric(levels(svm_pred2))[svm_pred2]

acc2 = 0
testlen2 = length(fdataset2$test$class)
for (i in 1:testlen2){
  if(svm_pred2[i] == fdataset2$test$class[i]){
    acc2 = acc2 + 1
  }
}
acc2 / testlen2