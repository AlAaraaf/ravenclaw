# Load the necessary libraries
library(keras)
library(reticulate)
library(tensorflow)
library(tiff)
library(magick)
library(imager)
library(jpeg)
library(png)
library(tidyverse)
library(fs)
library(tfautograph)
image_size <- 1000
splice_size <- 150
splice_number <- 10

##########------------q1---------------------------#######
# Define your data directory
data_dir <- "devnagari-bangla/"
class <- c("bangla", "devnagari")
new_dir <- "resize/"


## prepare training sets (here I resize to 1000x1000 to make resolution equal)
# Define a function to resize an image and save it

for (current_class in class){
  current_folder = paste(data_dir, paste(current_class,'/',sep = ''),sep = '')
  filelist = list.files(current_folder)
  for (current_file in filelist){
    input_dir = paste(current_folder, current_file, sep = '')
    img = image_read(input_dir)
    simg = image_scale(img,"1000x1000!")
    file_name <- basename(file_path_sans_ext(current_file, compression = TRUE))
    output_path <- file.path(paste(new_dir, current_class,sep = ''), file_name)
    image_write(simg, path = paste(output_path, ".", "png", sep = ""),format = 'PNG')
  }
}

## randomly sample n x n x 1 images from the original image
splice <- function(image, size) {
  repeat {
    pos_y <- sample(1:(dim(image)[1]-(size-1)), 1)
    pos_x <- sample(1:(dim(image)[2]-(size-1)), 1)
    candidate <- image[pos_y:(pos_y+(size-1)), pos_x:(pos_x+(size-1))]
    
    # keep selecting images until you find one that isn't mostly whitespace
    white <- length(which(round(unlist(candidate)) == 1))/(dim(candidate)[1]*dim(candidate)[2])
    
    # add fail safe in case valid images can't be found
    if (exists('attempts') == FALSE) {
      attempts <- 1
    } else {
      attempts <- attempts + 1
    }
    if (white < 0.90 | attempts > 20) {
      break
    }
  }
  candidate
}

## import bangla data
filepath_bangla <- list.files(file.path("resize", "bangla"), full.names = TRUE)
pb <- txtProgressBar(min = 0, max = length(filepath_bangla), initial = 0, style = 3) 
data_ban <- array(dim = c(length(filepath_bangla)*splice_number, splice_size, splice_size, 1))
for (n in 1:length(filepath_bangla)) {
  for (i in 1:splice_number) {
    data_ban[(n-1)*splice_number+i,,,] <- filepath_bangla[i] |> readPNG() |> splice(size = 150) |> round()
    rm(i)
  }
  setTxtProgressBar(pb, n)
  rm(n)
}

## import devnagari data
filepath_devnagari <- list.files(file.path("resize", "devnagari"), full.names = TRUE)
pb <- txtProgressBar(min = 0, max = length(filepath_devnagari), initial = 0, style = 3) 
data_dev <- array(dim = c(length(filepath_devnagari)*splice_number, splice_size, splice_size, 1))
for (n in 1:length(filepath_devnagari)) {
  for (i in 1:splice_number) {
    data_dev[(n-1)*splice_number+i,,,] <- filepath_devnagari[i] |> readPNG() |> splice(size = 150) |> round()
    rm(i)
  }
  setTxtProgressBar(pb, n)
  rm(n)
}

## randomly select 0.1percent images for testing (5 of each)
set.seed(777)
numbangla <- 0.1*length(filepath_bangla)*splice_number
numdevnagari <- 0.1*length(filepath_devnagari)*splice_number
test <- array(dim = c(numbangla+numdevnagari, dim(data_ban)[-1]))
split <- sample(1:dim(data_ban)[1], numbangla)
test[1:numbangla,,,] <- data_ban[split,,,]
data_ban_train <- array(dim = c(dim(data_ban)[1]-length(split), dim(data_ban)[-1]))
data_ban_train[1:dim(data_ban_train)[1],,,] <- data_ban[-split,,,]
split <- sample(1:dim(data_dev)[1], numdevnagari)
test[(numbangla+ 1):(numbangla+numdevnagari),,,] <- data_dev[split,,,]
data_dev_train <- array(dim = c(dim(data_dev)[1]-length(split), dim(data_dev)[-1]))
data_dev_train[1:dim(data_dev_train)[1],,,]  <- data_dev[-c(split),,,]
rm(split)

## combine data sets into training set
train <- array(dim = c(
  dim(data_ban_train)[1] + dim(data_dev_train)[1],
  dim(data_ban_train)[-1]))
train[1:(dim(data_ban_train)[1]),,,] <- data_ban_train
train[(dim(data_ban_train)[1]+1):(dim(train)[1]),,,] <- data_dev_train

## create labels
train_y <- c(rep(0, dim(data_ban_train)[1]), rep(1, dim(data_dev_train)[1]))
test_y <- c(rep(0, numbangla), rep(1, numdevnagari)) |> to_categorical()

## shuffle training set
set.seed(777)
shuffle <- sample(dim(train)[1])
train_y <- train_y[shuffle] |> to_categorical()
train[1:dim(train)[1],,,] <- train[shuffle,,,] 
rm(shuffle)


########### Model Architecture
build_convnet <- function(filters_list, kernel_size = c(3, 3), pool_size = c(2, 2),
                          input_shape, output_class, dropout = 0, initial_learning_rate = 0.1,
                          rotation = F, flipping = F, batch_normalization = F, separable_conv = F, resid_connection = F,
                          printmodel = T) {
  k_clear_session()
  layer_conv <- ifelse(separable_conv, layer_separable_conv_2d, layer_conv_2d)
  
  
  hidden <- input <- layer_input(shape = input_shape)
  
  if (rotation) {
    hidden <- layer_random_rotation(factor = 0.2)(hidden)
  }
  if (flipping) {
    hidden <- layer_random_flip(mode = "horizontal")(hidden)
    #hidden <- tf.keras.layers.RandomFlip("vertical")(hidden)
  }
  for (filter in filters_list) {
    resid <- hidden
    if (batch_normalization) {
      hidden <- layer_conv(filters = filter, kernel_size = kernel_size, padding = "same", use_bias = F)(hidden)
      hidden <- layer_batch_normalization()(hidden)
      hidden <- layer_activation_relu()(hidden)
    } else hidden <- layer_conv(filters = filter, kernel_size = kernel_size, padding = "same", activation = "relu")(hidden)
    hidden <- layer_max_pooling_2d(pool_size = pool_size, padding = "same")(hidden)
    
    if (resid_connection) {
      resid <- layer_conv_2d(filters = filter, kernel_size = c(1, 1), padding = "same", strides = pool_size)(resid)
      hidden <- layer_add(hidden, resid)
    }
    if (dropout > 0) hidden <- layer_dropout(rate = dropout)(hidden)
  }
  hidden <- layer_flatten()(hidden)
  
  if (batch_normalization) {
    hidden <- layer_dense(units = 32, use_bias = F)(hidden)
    hidden <- layer_batch_normalization()(hidden)
    hidden <- layer_activation_relu()(hidden)
  } else hidden <- layer_dense(units = 32, activation = "relu")(hidden)
  
  output <- layer_dense(units = output_class, activation = "sigmoid")(hidden)
  model <- keras_model(inputs = input, outputs = output)
  if (printmodel) summary(model, show_trainable = T)
  
  learning_rate <- learning_rate_schedule_exponential_decay(initial_learning_rate,
                                                            decay_steps = 5, decay_rate = 0.9, staircase = T)
  model %>% compile(
    loss = loss_binary_crossentropy(),
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = list("acc")
  )
  model
}


##################################
## run for 1(a)
####################################

filter_list <- list(2 ^ (3:4), 2 ^ (4:5) ,2 ^ (3:5), 
                    2 ^ (4:6),2 ^ (3:6), 2 ^ (4:7) , 
                    2 ^ (4:8),2 ^ (5:9), 2 ^ (4:9))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]]
  
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y[, 1, drop = F], epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y[, 1, drop = F])
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result11.csv")


##################################
## run for 1(b)_rotation
####################################
filter_list <- list(2 ^ (3:5), 
                    2 ^ (4:7) , 
                    2 ^ (4:8), 2 ^ (4:9))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list,rotation = T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = T, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y[, 1, drop = F], epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y[, 1, drop = F])
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result12_r.csv")

##################################
## run for 1(b)_flipping
####################################
filter_list <- list(2 ^ (3:5), 
                    2 ^ (4:7) , 
                    2 ^ (4:8), 2 ^ (4:9))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list, flipping = T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = T, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y[, 1, drop = F], epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y[, 1, drop = F])
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result12_f.csv")

##################################
## run for 1(c)
####################################
filter_list <- list( 2 ^ (4:8), 2 ^ (4:9))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list, resid_connection =T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = T, 
                         printmodel = T)
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y[, 1, drop = F], epochs = 20, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y[, 1, drop = F])
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result13_r.csv")


####################################
## run for 1(d)
####################################

### best model with filter sequence c(16, 32, 64, 128, 256), and consider batch normalization.
model <- build_convnet(filters_list = c(16,32,64,128,256), kernel_size = c(3, 3), pool_size = c(2, 2),
                       input_shape = dim(train)[-1], output_class = 1,
                       dropout = 0, initial_learning_rate = 0.001, 
                       rotation = F, flipping = F, batch_normalization = T, 
                       separable_conv = F, resid_connection = F, 
                       printmodel = T)

tf_read_image <- function(path, format = "image", resize = NULL, ...) {
  img <- path |>
    tf$io$read_file() |>
    tf$io[[paste0("decode_", format)]](...)
  if (!is.null(resize))
    img <- img |>
      tf$image$resize(as.integer(resize)) ## call tf module function with integers using as.integer().
  img
}

display_image_tensor <- function(x,..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {
  if(!is.null(plot_margins))
    par(mar = plot_margins)
  x %>%
    as.array() %>%
    drop() %>%
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
}


img <- train[1,,,]
img <- writeJPEG(img,"1e.jpeg")

img_path <-"1e.jpeg"
img_tensor <- img_path |> tf_read_image(resize = c(150, 150))
display_image_tensor(img_tensor)


conv_layer_s3_classname <- class(layer_conv_2d(NULL, 1, 1))[1]
## Make dummy conv and pooling layers to determine what the S3 classname is. This is generally a long
## string like "keras.layers.convolutional.Conv2D", but since is can change between Tensorflow versions,
## best not to hardcode it.
pooling_layer_s3_classname <- class(layer_max_pooling_2d(NULL))[1]
is_conv_layer <- function(x) inherits(x, conv_layer_s3_classname)
is_pooling_layer <- function(x) inherits(x, pooling_layer_s3_classname)
layer_outputs <- list()
for (layer in model$layers)
  if (is_conv_layer(layer) || is_pooling_layer(layer))
    layer_outputs[[layer$name]] <- layer$output
## Extract the outputs of all Conv2D and MaxPooling2D layers and put them in a named list
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
activations <- activation_model |>
  predict(img_tensor[tf$newaxis, , , ])

first_layer_activation <- activations[[names(layer_outputs)[4]]]
dim(first_layer_activation)
plot_activations <- function(x, ...) {
  x <- as.array(x)
  if(sum(x) == 0)
    return(plot(as.raster("gray")))
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

###Visualizing the fifth channel
par(mfrow=c(1,1), mar = rep(0.01,4))
plot_activations(first_layer_activation[, , , 5])

###Visualizing every channel in every intermediate activation
for (layer_name in names(layer_outputs)) {
  layer_output <- activations[[layer_name]]
  n_features <- dim(layer_output) %>% tail(1) 
  par(mfrow = n2mfrow(n_features, asp = 1.75),
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))
  for (j in 1:n_features)
    plot_activations(layer_output[, , , j])
  title(main = layer_name, outer = TRUE)
}

####################################
## run for 1(e)
####################################

### Loading the Xception network with pretrained weights

img_path <-"1e.jpeg"
img_tensor <- img_path |> tf_read_image(resize = c(150, 150))
preprocessed_img <- (img_tensor[tf$newaxis, , , ] /255 ) * 2 - 1
preds <- predict(model, preprocessed_img)
str(preds)
imagenet_decode_predictions(preds, top=3)[[1]]

layer_names <- sapply(model$layers, `[[`, "name")
layer_name <- "max_pooling2d_4"
idx <- match(layer_name, layer_names)

model_prev_lastlayer <- get_layer(model, layer_name)
model_prev <- keras_model(model$inputs, model_prev_lastlayer$output)

x <- model_next_input <- layer_input(batch_shape = model_prev_lastlayer$output$shape)
for (. in layer_names[-(1:idx)]) x <- get_layer(model, .)(x)
model_next <- keras_model(model_next_input, x)


with (tf$GradientTape() %as% tape, {
  model_prev_output <- model_prev(preprocessed_img)
  tape$watch(model_prev_output)
  ## Compute activations of the last conv layer and make the tape watch it.
  preds <- model_next(model_prev_output)
  top_pred_index <- tf$argmax(preds[1, ])
  top_class_channel <- preds[, top_pred_index, style = "python"]
  ## Retrieve the activation channel corresponding to the top predicted class.
})
grads <- tape$gradient(top_class_channel, model_prev_output)

pooled_grads <- mean(grads, axis = 1:3, keepdims = TRUE)



heatmap <-
  ## Multiply each channel in output of last convolutional layer by importance of this channel.
  (model_prev_output * pooled_grads) |>
  ## grads and last_conv_layer_output have same shape, (1, 10, 10, 2048) and pooled_grads has shape (1, 1, 1, 2048).
  mean(axis = -1) %>% ## Shape: (1, 10, 10). Channel-wise mean of the resulting feature map is heatmap
  ## of class activation. Note use of %>%, not |>
  .[1, , ] ## Drop batch dim; output shape: (10, 10)
par(mar = c(0, 0, 0, 0))
plot_activations(heatmap)


pal <- hcl.colors(256, palette = "Spectral", alpha = .4, rev = TRUE)
heatmap <- as.array(heatmap)
heatmap[] <- pal[cut(heatmap, 256)]
heatmap <- as.raster(heatmap)
img <- tf_read_image(img_path, resize = NULL) ## Load the original image, without resizing this time.
display_image_tensor(img)
rasterImage(heatmap, 0, 0, ncol(img), nrow(img), interpolate = FALSE)

#####-----------------------------2(a)-----------------------------###

#### wrapped functions ####
read.in.all.file <- function(folders, class){
  # INPUT
  # class - the class vector including all required class for classification
  
  # OUTPUT
  # filelist - the dataframe [filedir, class]
  filelist = data.frame()
  for (i in 0:(length(class) - 1)) {
    current_dir =  paste(folders, paste(class[i+1],'/',sep = ''),sep = '')
    allfiles = list.files(current_dir)
    allfiles = unlist(lapply(allfiles, function(x) {
      paste(current_dir, x, sep = '')
    }))
    current_filelist = data.frame(filedir = allfiles,
                                  class = rep(i, length(allfiles)))
    filelist = rbind(filelist, current_filelist)
  }
  
  return(filelist)
}


make.dataset <- function(filelist, p=100, img_width, img_height, imagetype){
  # INPUT
  # filelist - the dataframe [filedir, class]
  # p - the number test sample
  # OUTPUT
  # the dataset list [train[data, tag], test[data, tag]]
  
  n = dim(filelist)[1]
  filelist = filelist[sample(dim(filelist)[1], n),]
  test_id = sample(n, p)
  train_file = filelist[-test_id,]
  test_file = filelist[test_id,]
  
  dataset = list()
  trainset = list()
  trainset$data = array(0, dim = c(dim(train_file)[1], img_width, img_height, 3))
  testset = list()
  testset$data = array(0, dim = c(dim(test_file)[1], img_width, img_height, 3))
  
  for (i in 1:dim(train_file)[1]){
    current_file = train_file[i,]
    if(imagetype==1){
      current_img = readTIFF(current_file$filedir)
    }
    if(imagetype==0){
      current_img = readJPEG(current_file$filedir)
    }
    trainset$data[i,,,] <- current_img
  }
  trainset$class = train_file$class
  
  for (i in 1:dim(test_file)[1]){
    current_file = test_file[i,]
    if(imagetype==1){
      current_img = readTIFF(current_file$filedir)
    }
    if(imagetype==0){
      current_img = readJPEG(current_file$filedir)
    }
    testset$data[i,,,] <- current_img
  }
  testset$class = test_file$class
  
  dataset$train <-  trainset
  dataset$test <-  testset
  
  return(dataset)
}


# Define your data directory
data_dir_2 <- "Pomegranate/"
class_2 <- c("disease", "health")
new_dir_2 <- "resize2/"


# Define a function to resize an image and save it
# for (current_class in class_2){
#   current_folder = paste(data_dir_2, paste(current_class,'/',sep = ''),sep = '')
#   filelist = list.files(current_folder)
#   for (current_file in filelist){
#     input_dir = paste(current_folder, current_file, sep = '')
#     output_dir = paste(new_dir_2, paste(current_class,'/',current_file,sep = ''),sep = '')
#     img = image_read(input_dir)
#     simg = image_scale(img,"256x256!")
#     image_write(simg, output_dir)
#   }
# }

filelist_2 <- read.in.all.file(new_dir_2,class_2)
p = round(0.1*length(filelist_2$filedir),0)
dataset_2 <- make.dataset(filelist_2, p, img_width = 256, img_height = 256,imagetype=0)

dataset <- dataset_2



##################################
## run for 2(a)
####################################
filter_list <- list(2 ^ (3:4), 2 ^ (4:5),
                    2 ^ (3:5), 2 ^ (4:6), 2 ^ (3:6), 2 ^ (4:7),
                    2 ^ (5:8), 2 ^ (3:7), 2 ^ (3:8))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]]
  
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(dataset$train$data)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(dataset$train$data, dataset$train$class, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(dataset$test$data, dataset$test$class)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result21.csv")

##################################
## run for 2(b)_rotation
####################################
filter_list <- list( 2 ^ (4:5), 2 ^ (4:7),
                     2 ^ (5:8), 2 ^ (3:7))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list,rotation = T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(dataset$train$data)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = T, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(dataset$train$data, dataset$train$class, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(dataset$test$data, dataset$test$class)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result22_r.csv")

##################################
## run for 2(b)_flipping
####################################
filter_list <- list( 2 ^ (4:5), 2 ^ (4:7),
                     2 ^ (5:8), 2 ^ (3:7))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list, flipping = T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(dataset$train$data)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = T, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(dataset$train$data, dataset$train$class, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(dataset$test$data, dataset$test$class)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result22_f.csv")


##################################
## run for 2(c)
####################################
filter_list <- list( 2 ^ (4:5), 2 ^ (4:7),
                     2 ^ (5:8), 2 ^ (3:7))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(dataset$train$data)[-1], output_class = 1,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = T, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(dataset$train$data, dataset$train$class, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(dataset$test$data, dataset$test$class)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result23_T.csv")


####################################
## run for 2(d)
####################################

model <- build_convnet(filters_list = c(16,32), kernel_size = c(3, 3), pool_size = c(2, 2),
                       input_shape = dim(dataset$train$data)[-1], output_class = 1,
                       dropout = 0, initial_learning_rate = 0.001, 
                       rotation = T, flipping = F, batch_normalization = F, 
                       separable_conv = F, resid_connection = F, 
                       printmodel = T)

tf_read_image <- function(path, format = "image", resize = NULL, ...) {
  img <- path |>
    tf$io$read_file() |>
    tf$io[[paste0("decode_", format)]](...)
  if (!is.null(resize))
    img <- img |>
      tf$image$resize(as.integer(resize)) ## call tf module function with integers using as.integer().
  img
}

display_image_tensor <- function(x,..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {
  if(!is.null(plot_margins))
    par(mar = plot_margins)
  x %>%
    as.array() %>%
    drop() %>%
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
}

img_path <-"D://iowa state//STAT590s3//hw3//Pomegranate//disease//0020_0001.JPG"
img_tensor <- img_path |> tf_read_image(resize = c(256, 256))
display_image_tensor(img_tensor)


conv_layer_s3_classname <- class(layer_conv_2d(NULL, 1, 1))[1]
## Make dummy conv and pooling layers to determine what the S3 classname is. This is generally a long
## string like "keras.layers.convolutional.Conv2D", but since is can change between Tensorflow versions,
## best not to hardcode it.
pooling_layer_s3_classname <- class(layer_max_pooling_2d(NULL))[1]
is_conv_layer <- function(x) inherits(x, conv_layer_s3_classname)
is_pooling_layer <- function(x) inherits(x, pooling_layer_s3_classname)
layer_outputs <- list()
for (layer in model$layers)
  if (is_conv_layer(layer) || is_pooling_layer(layer))
    layer_outputs[[layer$name]] <- layer$output
## Extract the outputs of all Conv2D and MaxPooling2D layers and put them in a named list
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
activations <- activation_model |>
  predict(img_tensor[tf$newaxis, , , ])

first_layer_activation <- activations[[names(layer_outputs)[4]]]
dim(first_layer_activation)
plot_activations <- function(x, ...) {
  x <- as.array(x)
  if(sum(x) == 0)
    return(plot(as.raster("gray")))
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

###Visualizing the fifth channel
par(mfrow=c(1,1), mar = rep(0.01,4))
plot_activations(first_layer_activation[, , , 5])



###Visualizing every channel in every intermediate activation
for (layer_name in names(layer_outputs)) {
  layer_output <- activations[[layer_name]]
  n_features <- dim(layer_output) %>% tail(1) 
  par(mfrow = n2mfrow(n_features, asp = 1.75),
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))
  for (j in 1:n_features)
    plot_activations(layer_output[, , , j])
  title(main = layer_name, outer = TRUE)
}


####################################
## run for 2(e)
####################################

img_path <-"D://iowa state//STAT590s3//hw3//Pomegranate//disease//0020_0001.JPG"
img_tensor <- img_path |> tf_read_image(resize = c(256, 256))
preprocessed_img <- (img_tensor[tf$newaxis, , , ] / 255) * 2 - 1
preds <- predict(model, preprocessed_img)
str(preds)
imagenet_decode_predictions(preds, top=3)[[1]]


layer_names <- sapply(model$layers, `[[`, "name")
layer_name <- "max_pooling2d_1"
idx <- match(layer_name, layer_names)

model_prev_lastlayer <- get_layer(model, layer_name)
model_prev <- keras_model(model$inputs, model_prev_lastlayer$output)

x <- model_next_input <- layer_input(batch_shape = model_prev_lastlayer$output$shape)
for (. in layer_names[-(1:idx)]) x <- get_layer(model, .)(x)
model_next <- keras_model(model_next_input, x)


with (tf$GradientTape() %as% tape, {
  model_prev_output <- model_prev(preprocessed_img)
  tape$watch(model_prev_output)
  ## Compute activations of the last conv layer and make the tape watch it.
  preds <- model_next(model_prev_output)
  top_pred_index <- tf$argmax(preds[1, ])
  top_class_channel <- preds[, top_pred_index, style = "python"]
  ## Retrieve the activation channel corresponding to the top predicted class.
})

grads <- tape$gradient(top_class_channel, model_prev_output)

pooled_grads <- mean(grads, axis = c(2, 1, 3), keepdims = TRUE)


heatmap <-
  ## Multiply each channel in output of last convolutional layer by importance of this channel.
  (model_prev_output * pooled_grads) |>
  ## grads and last_conv_layer_output have same shape, (1, 10, 10, 2048) and pooled_grads has shape (1, 1, 1, 2048).
  mean(axis = -1) %>% ## Shape: (1, 10, 10). Channel-wise mean of the resulting feature map is heatmap
  ## of class activation. Note use of %>%, not |>
  .[1, , ] ## Drop batch dim; output shape: (10, 10)
par(mar = c(0, 0, 0, 0))
plot_activations(heatmap)


pal <- hcl.colors(256, palette = "Spectral", alpha = .4, rev = TRUE)
heatmap <- as.array(heatmap)
heatmap[] <- pal[cut(heatmap, 256)]
heatmap <- as.raster(heatmap)
img <- tf_read_image(img_path, resize = NULL) ## Load the original image, without resizing this time.
display_image_tensor(img)
rasterImage(heatmap, 0, 0, ncol(img), nrow(img), interpolate = FALSE)
## Superimpose the heatmap over the original image, with the heatmap at 40% opacity. We pass ncol(img)
## and nrow(img) so that the heatmap, which has fewer pixels, is drawn to match the size of the original
#


################ Q3 ###############

library(tidyverse)
library(RNifti)  # readNifti(), channels()
library(rayshader)  # resize_matrix()
library(keras)
library(deepviz)  # plot_model() // devtools::install_github("andrie/deepviz")
library(imageseg)  # u_net() // remotes::install_github("jniedballa/imageseg")
library(caret)  # confusionMatrix()

greyscale <- function(RGB) {
  if (length(dim(RGB)) > 2) {
    RGB[,,,1]*0.3 + RGB[,,,2]*0.59 + RGB[,,,3]*0.11
  } else {
    RGB[,,,1]*NA  # image
  }
}

################ Resize and Cleanse Data ###############

res <- 128  # resized resolution

current_dir <- getwd()
output_dir <- file.path(current_dir, "AeroPathResize")
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}
pb <- txtProgressBar(min = 0, max = 27*3, style = 3)
ipb <- 1

for (patient in 1:27) {
  input_dir <- file.path(current_dir, "AeroPath", patient)
  files <- paste0(patient, c("_CT_HR.nii.gz", "_CT_HR_label_airways.nii.gz", "_CT_HR_label_lungs.nii.gz"))
  
  for (file in files) {
    file_path <- file.path(input_dir, file)
    image <- readNifti(file = file_path)
    image <- image %>% channels() %>% greyscale() / 255
    
    depth <- dim(image)[3]
    image_resize <- array(dim = c(depth, res, res))
    for (i in 1:depth) {
      image_resize[i,,] <- image[,,i] %>% resize_matrix(width = res, height = res, method = 'bilinear')
    }
    
    if (grepl("airways", file)) {
      output_file <- paste0("y", patient, "_air.rds")
    } else if (grepl("lungs", file)) {
      output_file <- paste0("y", patient, "_lung.rds")
    } else {
      output_file <- paste0("x", patient, ".rds")
    }
    
    saveRDS(image_resize, file = file.path(output_dir, output_file))
    setTxtProgressBar(pb, ipb)
    ipb <- ipb + 1
  }
}

rm(list = ls(all = TRUE))

################ Load Resized Data ###############

res <- 128
n_sample_each <- 200
n_sample <- 26*n_sample_each

current_dir <- getwd()
input_dir <- file.path(current_dir, "AeroPathResize")

x <- array(dim = c(n_sample, res, res))
y0 <- array(data = 1, dim = c(n_sample, res, res))  # neither
y1 <- array(dim = c(n_sample, res, res))  # lung
y2 <- array(dim = c(n_sample, res, res))  # airway
y <- array(dim = c(n_sample, res, res, 3))  # concatenate neither, lung and airway

pb <- txtProgressBar(min = 0, max = 26, style = 3)
set.seed(0)
for (patient in 1:26) {
  x_path <- file.path(input_dir, paste0("x", patient, ".rds"))
  y_lung_path <- file.path(input_dir, paste0("y", patient, "_lung.rds"))
  y_air_path <- file.path(input_dir, paste0("y", patient, "_air.rds"))
  
  x_tmp <- readRDS(x_path)
  y_lung_tmp <- readRDS(y_lung_path)
  y_air_tmp <- readRDS(y_air_path)
  
  depth <- dim(x_tmp)[1]
  slices <- sample(1:depth, size = n_sample_each, replace = FALSE)  # make replace = TRUE if n_sample_each > min(depth) = 265
  
  x[1:n_sample_each+(patient-1)*n_sample_each,,] <- x_tmp[slices,,]
  y1[1:n_sample_each+(patient-1)*n_sample_each,,] <- y_lung_tmp[slices,,]
  y2[1:n_sample_each+(patient-1)*n_sample_each,,] <- y_air_tmp[slices,,]
  
  setTxtProgressBar(pb, patient)
}

y1[y1 > 0] <- 1
y2[y2 > 0] <- 1
y2[y1 == y2] <- 0  # if indetermined, set it as lung
y0[y0 == y1] <- 0  # if lung, set neither as 0
y0[y0 == y2] <- 0  # if airway, set neither as 0

y[,,,1] <- y0  # neither
y[,,,2] <- y1  # lung
y[,,,3] <- y2  # airway

test_idx <- sample(1:n_sample, size = round(0.2*n_sample))
train_idx <- sample(setdiff(1:n_sample, test_idx))

x_train <- x[train_idx,,]
y_train <- y[train_idx,,,]
x_test <- x[test_idx,,]
y_test <- y[test_idx,,,]

rm(x, y, y0, y1, y2)
rm(pb, depth, input_dir, n_sample, patient, slices, test_idx, train_idx, x_path, x_tmp, y_air_path, y_air_tmp, y_lung_path, y_lung_tmp)

################ Fitting ###############

#### U-Net Architecture ####
model <- imageseg::u_net(
  res,
  res,
  grayscale = TRUE,
  blocks = 2,
  n_class = 3
)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)
# summary(model)
model %>% deepviz::plot_model()

set.seed(0)
epochs <- 10

history <- model %>% fit(
  x = x_train,
  y = y_train,
  batch_size = 32,
  epochs = epochs,
  validation_split = 0.3,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 5,
      restore_best_weights = TRUE
    )
  )
)

################ Evaluate ###############

evaluate(model, x_test, y_test)

################ 27th Last Image ###############

input_dir <- file.path(current_dir, "AeroPathResize")

x27 <- array(dim = c(n_sample_each, res, res))
y27_0 <- array(data = 1, dim = c(n_sample_each, res, res))
y27_lung <- array(dim = c(n_sample_each, res, res))
y27_air <- array(dim = c(n_sample_each, res, res))
y27 <- array(dim = c(n_sample_each, res, res, 3))

set.seed(0)
patient <- 27

x_path <- file.path(input_dir, paste0("x", patient, ".rds"))
y_lung_path <- file.path(input_dir, paste0("y", patient, "_lung.rds"))
y_air_path <- file.path(input_dir, paste0("y", patient, "_air.rds"))

x_tmp <- readRDS(x_path)
y_lung_tmp <- readRDS(y_lung_path)
y_air_tmp <- readRDS(y_air_path)

depth <- dim(x_tmp)[1]
slices <- sample(1:depth, size = n_sample_each)

x27[1:n_sample_each,,] <- x_tmp[slices,,]
y27_lung[1:n_sample_each,,] <- y_lung_tmp[slices,,]
y27_air[1:n_sample_each,,] <- y_air_tmp[slices,,]

y27_lung[y27_lung > 0] <- 1
y27_air[y27_air > 0] <- 1
y27_air[y27_lung == y27_air] <- 0  # if indetermined, set it as lung
y27_0[y27_0 == y27_lung] <- 0  # if lung, set neither as 0
y27_0[y27_0 == y27_air] <- 0  # if airway, set neither as 0

y27[,,,1] <- y27_0  # neither
y27[,,,2] <- y27_lung  # lung
y27[,,,3] <- y27_air  # airway

evaluate(model, x27, y27)

s <- 1  # slice
x27[s,,] %>% as.raster() %>% plot()  # train image
y27[s,,,1] %>% as.raster() %>% plot()  # neither
y27[s,,,2] %>% as.raster() %>% plot()  # lung
y27[s,,,3] %>% as.raster() %>% plot()  # airway

# out <- cbind(x27[s,,], y27[s,,,1], y27[s,,,2], y27[s,,,3])
# png::writePNG(out, paste0(s,'.png'))

y27_pred <- predict(model, x27)

s <- 1
y27_pred[s,,,1] %>% as.raster() %>% plot()
y27_pred[s,,,2] %>% as.raster() %>% plot()
y27_pred[s,,,3] %>% as.raster() %>% plot()

# out <- cbind(y27_pred[s,,,1], y27_pred[s,,,2], y27_pred[s,,,3])
# png::writePNG(out, paste0(s,"pred.png"))

y27_categorical_true <- apply(y27, c(1,2,3), function(x) {x %*% c(1,2,3)})
y27_categorical_pred <- apply(y27_pred, c(1,2,3), which.max)
confusionMatrix(as.factor(y27_categorical_true), as.factor(y27_categorical_pred))
