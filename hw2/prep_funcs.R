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

convert_onehot <- function(dataset){
  dataset$train$class = to_categorical(dataset$train$class)
  dataset$test$class = to_categorical(dataset$test$class)

  return(dataset)
}
