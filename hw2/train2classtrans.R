source('./prep_funcs.R')
library(reticulate)
use_condaenv('connectr')
library(tensorflow)
tf <- import('tensorflow')
library(keras)
tf$config$run_functions_eagerly(T)
library(jpeg)

# set dataset
classlist <- c('cat','dog')
folders = c('./dataset2/train/','./dataset2/val/')

# model hyperparameters
layerlist = c(2,4,8,12,16)
epochs = c(10,10,10,20,20)
unitlists =list()
unitlists[[1]] <- c(32,16) # 2 layers
unitlists[[2]] <- c(32,32,16,16) # 4 layers
unitlists[[3]] <- c(rep(32,5),rep(16,3)) # 8 layers
unitlists[[4]] <- c(rep(32,3),rep(32,6), rep(16,3)) # 12 layers
unitlists[[5]] <- c(rep(32,3),rep(32,9), rep(16,4)) # 16 layers

checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = 0

# read in data
filelist = read.in.all.file(folders, classlist)
dataset = make.dataset(dim(filelist)[1], filelist, 0.75, 20, 64)

# model fit
for (lr in lrlist){
  for (i in 1:5){
    model_name = paste(checkpoint_folder, '2class_', i,'layer', '_',lr,sep = '')
    model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                save_weights_only = T,
                                                save_best_only = T,
                                                monitor = 'val_accuracy',
                                                mode = 'max')
    
    model <- construct_fc_model(layerlist[i], unitlists[[i]], dropout = dropout_rate)
    model %>% keras::compile(optimizer = optimizer_sgd(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    historys[[i]] <- model |>
      fit(x = dataset$train$data,
          y = dataset$train$class,
          epochs = epochs[i], batch_size = 64, validation_split = 0.2,
          callbacks = list(model_cp))
  }
}

# model evaluate
for (lr in lrlist){
  cat("current lr:", lr, '\n')
  for (i in 1:5){
    model_name = paste(checkpoint_folder, '2class_', i,'layer','_',lr, sep = '')
    model <- construct_fc_model(layerlist[i], unitlists[[i]], dropout = dropout_rate)
    
    model %>% keras::compile(optimizer = optimizer_sgd(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    model %>% load_model_weights_tf(filepath = model_name)
    model %>% evaluate(dataset$test$data, dataset$test$class, batch_size = 64, verbose = '2')
  }
}
