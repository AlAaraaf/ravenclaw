# Load the necessary libraries
library(keras)
library(reticulate)
library(tensorflow)
library(tiff)
library(magick)
library(imager)
library(jpeg)
library(tidyverse)
library(fs)
library(tfautograph)
image_size <- 1000
splice_size <- 150
splice_number <- 10


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
  
  output <- layer_dense(units = output_class, activation = "softmax")(hidden)
  model <- keras_model(inputs = input, outputs = output)
  if (printmodel) summary(model, show_trainable = T)
  
  learning_rate <- learning_rate_schedule_exponential_decay(initial_learning_rate,
                                                            decay_steps = 5, decay_rate = 0.9, staircase = T)
  model %>% compile(
    loss = loss_categorical_crossentropy(),
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = list("acc")
  )
  model
}


##################################
## run for 1(a)
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
                         input_shape = dim(train)[-1], output_class = 2,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res)


##################################
## run for 1(b)_rotation
####################################
filter_list <- list( 2 ^ (4:5), 2 ^ (4:7),
                     2 ^ (5:8), 2 ^ (3:7))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list,rotation = T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 2,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = T, flipping = F, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) 

##################################
## run for 1(b)_flipping
####################################
filter_list <- list( 2 ^ (4:5), 2 ^ (4:7),
                     2 ^ (5:8), 2 ^ (3:7))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list, flipping = T)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 2,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = T, batch_normalization = F, 
                         separable_conv = F, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res)

##################################
## run for 1(c)
####################################
filter_list <- list( 2 ^ (4:5), 2 ^ (4:7),
                     2 ^ (5:8), 2 ^ (3:7))

pars <- expand.grid(lr_initial = 0.001, filter = filter_list)
res <- NULL
for (i in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[i]
  filters = pars$filter[[i]] 
  model <- build_convnet(filters_list = filters, kernel_size = c(3, 3), pool_size = c(2, 2),
                         input_shape = dim(train)[-1], output_class = 2,
                         dropout = 0, initial_learning_rate = lr_initial, 
                         rotation = F, flipping = F, batch_normalization = F, 
                         separable_conv = T, resid_connection = F, 
                         printmodel = T)
  
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(train, train_y, epochs = 50, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  results <- model %>% evaluate(test, test_y)
  names(results) <- c("test_loss", "test_acc")
  res <- rbind(res, cbind(last(data.frame(history$metrics)),t(results)))
}

cbind(pars, res) %>% mutate(filter = as.character(filter)) %>%
  write.csv("result13_T.csv")

####################################
## run for 1(d)
####################################

### best model with filter sequence 8*2,16*2,32*2,64*2,128*2, and consider batch normalization.
model <- build_convnet(filters_list = c(16,32,64,128,256), kernel_size = c(3, 3), pool_size = c(2, 2),
                       input_shape = dim(train)[-1], output_class = 2,
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

img_path <-"D://iowa state//STAT590s3//hw3//resize//bangla//p_ben_0001.png"
img_tensor <- img_path |> tf_read_image(resize = c(1000, 1000))
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
model <- application_xception(weights = "imagenet")
img_path <-"D://iowa state//STAT590s3//hw3//resize//bangla//p_ben_0001.png"
img_tensor <- img_path |> tf_read_image(resize = c(299, 299))
preprocessed_img <- img_tensor[tf$newaxis, , , ] %>% xception_preprocess_input()
preds <- predict(model, preprocessed_img)
str(preds)
imagenet_decode_predictions(preds, top=3)[[1]]