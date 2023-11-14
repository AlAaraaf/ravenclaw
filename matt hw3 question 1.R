#### Libraries ####
library(magick) # image_read(), image_scale(), image_write()
library(png) # readPNG()
library(caret) # createDataPartition()
library(keras)
library(ggplot2)
image_size <- 1000
splice_size <- 150
splice_number <- 10

## prepare training sets (here I resize to 1000x1000 to make resolution equal)
resize <- function(size, wd) {
  setwd(wd)
  for (i in 1:length(list.files())) {
    setwd(wd)
    temp <- image_read(list.files()[i])
    temp <- image_scale(temp, paste0(size, '!')) # ! = ignore aspect ratio
    setwd(paste0(wd, '_reduced'))
    image_write(image = temp, path = paste0(i, '.png'), format = 'PNG')
  }
}
resize(size = paste0(image_size, 'x', image_size), wd = 'C:/Users/Matt/Desktop/STAT 590B/hw3/bangla')
resize(size = paste0(image_size, 'x', image_size), wd = 'C:/Users/Matt/Desktop/STAT 590B/hw3/devnagari')

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
setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/bangla_reduced')
pb <- txtProgressBar(min = 0, max = length(list.files()), initial = 0, style = 3) 
data_ban <- array(dim = c(length(list.files())*splice_number, splice_size, splice_size, 1))
for (n in 1:length(list.files())) {
  for (i in 1:splice_number) {
    data_ban[(n-1)*splice_number+i,,,] <- list.files()[i] |> readPNG() |> splice(size = 150) |> round()
    rm(i)
  }
  setTxtProgressBar(pb, n)
  rm(n)
}

## import devnagari data
setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/devnagari_reduced')
pb <- txtProgressBar(min = 0, max = length(list.files()), initial = 0, style = 3) 
data_dev <- array(dim = c(length(list.files())*splice_number, splice_size, splice_size, 1))
for (n in 1:length(list.files())) {
  for (i in 1:splice_number) {
    data_dev[(n-1)*splice_number+i,,,] <- list.files()[i] |> readPNG() |> splice(size = 150) |> round()
    rm(i)
  }
  setTxtProgressBar(pb, n)
  rm(n)
}

## randomly select 10 images for testing (5 of each)
set.seed(777)
test <- array(dim = c(10*splice_number, dim(data_ban)[-1]))
split <- sample(1:dim(data_ban)[1], 10*splice_number/2)
test[1:(10*splice_number/2),,,] <- data_ban[split,,,]
data_ban <- data_ban[-c(split),,,]
split <- sample(1:dim(data_dev)[1], 10*splice_number/2)
test[(10*splice_number/2 + 1):(10*splice_number),,,] <- data_dev[split,,,]
data_dev <- data_dev[-c(split),,,]
rm(split)

## combine data sets into training set
train <- array(dim = c(
  dim(data_ban)[1] + dim(data_dev)[1],
  dim(data_ban)[-1],
  1))
train[1:(dim(data_ban)[1]),,,] <- data_ban
train[(dim(data_ban)[1]+1):(dim(train)[1]),,,] <- data_dev

## create labels
train_y <- c(rep(0, dim(data_ban)[1]), rep(1, dim(data_dev)[1]))
test_y <- c(rep(0, 10*splice_number/2), rep(1, 10*splice_number/2)) |> to_categorical()

## shuffle training set
set.seed(777)
shuffle <- sample(dim(train)[1])
train_y <- train_y[shuffle] |> to_categorical()
train <- train[shuffle,,,] 
rm(shuffle)


#### Train Model ####
model <- keras_model_sequential()
model |> 
  layer_conv_2d(filters = 8*2, kernel_size = c(3,3), input_shape = c(dim(train)[-1], 1)) |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_dropout(0.15) |> # these intermediate dropout layers dramatically increase val accuracy
  layer_normalization() |> 
  
  layer_conv_2d(filters = 16*2, kernel_size = c(3,3), activation = 'relu') |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_dropout(0.15) |> 
  layer_normalization() |> 
  
  layer_conv_2d(filters = 32*2, kernel_size = c(3,3), activation = 'relu') |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  layer_dropout(0.15) |> 
  layer_normalization() |>
  
  layer_flatten() |> 
  layer_dense(units = 32*2, activation = 'relu') |> 
  layer_normalization() |> 
  
  layer_dropout(0.5) |> 
  layer_dense(units = 2, activation = 'softmax')

## compile
model |> compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

## fit (accuracy doesn't improve on training set, but does improve for validation)
history <- model |> fit(
  x = train,
  y = train_y,
  batch_size = 32,
  epochs = 50,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_accuracy",
      patience = 5,
      restore_best_weights = TRUE))
)
evaluate(model, test, test_y)

