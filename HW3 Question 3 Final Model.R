#### README ####
  # Based off this blog post: https://forloopsandpiepkicks.wordpress.com/2021/04/09/image-segmentation-in-r-automatic-background-removal-like-in-a-zoom-conference/
  # The majority of this document is me reading and formatting data 
    # ydim and xdim = [n greyscale images, x resolution, y resolution]
      # 128x128 is a good balance between sufficient resolution and reasonable training time
      # resize_matrix() from package rayshader can be used to resize a [x, y, z] matrix into any size without having to export a new image like in magick
  # Steps for formatting data and training model for image segmentation:
      # 1)  Import 512 * 512 * N images (I already resized to 256*256 in another document)
      # 2)  Convert RGB to greyscale (already done in another document; RGB is meaningless for these images)
      # 3)  Resize to desired resolution (anything above 64*64)
      # 4)  Randomly sample a couple hundred images to train on (from each of the 26 files)
      # 5)  Creating training / test split (0.8, 0.2)
      # 6)  Load model architecture (I use U-Net, loaded from package imageseg)
      # 7)  Compile model with appropriate loss (Dice Loss for segmentation; binary crossentropy doesn't work for pixel-wise accuracy)
        # Dice loss function imported from package platypus
      # 8)  Train model using U-Net architecture (10 epochs for 128*128, 20 epochs for 64*64)
      # 9)  Evaluate model using Dice Coefficient (1 = 100% pixel accuracy, 0 = 0%)
      # 10) Plot true vs predicted segmentation of image 27 and save as .gif


#### Libraries ####
library(rayshader) # resize_matrix()
library(raster) # as.raster()
library(caret) # createDataPartition()
library(keras); keras_model_sequential() # this just initializes keras right at the start
library(deepviz) # plot_model() // devtools::install_github("andrie/deepviz")
library(imageseg) # u_net() // remotes::install_github("jniedballa/imageseg")
library(platypus) # loss_dice(), metric_dice_coeff() // remotes::install_github("maju116/platypus")


#### Settings ####
epochs <- 5 # 5 is sufficient for 128x128 and 64x64
res <- 256 # default to 256 (native image size), can be reduced to train on lower resolution images
sample <- 500 # number of images to sample along z-axis (100-200 is enough to learn training and test set, but not 27th inage)
images <- 26 # number of images to train on // range = [1, 26]


#### Import data (256x256; list of 26 Nifti images) ####
setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/aeropath resized')
x <- readRDS('x_resize.rds')
y_lung <- readRDS('y_lung_resize.rds')
y_air <- readRDS('y_air_resize.rds')

## randomly samples N images across z axis from the original 'stack' of images from scan
dice <- function (image_x, image_y1, image_y2, sample, res) {
  output <- list()
  temp_x <- array(dim = c(sample, res, res))
  temp_y1 <- array(dim = c(sample, res, res))
  temp_y2 <- array(dim = c(sample, res, res))
  
  set <- sample(1:length(image_x), sample, replace = TRUE)
  for (i in 1:sample) {
    slice <- set[sample]
    candidate_x <- image_x[[slice]]
    candidate_y1 <- image_y1[[slice]]
    candidate_y2 <- image_y2[[slice]]
    if (nrow(candidate_x) != res | ncol(candidate_x) != res) {
      candidate_x <- candidate_x |> resize_matrix(width = res, height = res, method = 'bilinear')
      candidate_y1 <- candidate_y1 |> resize_matrix(width = res, height = res, method = 'bilinear')
      candidate_y2 <- candidate_y2 |> resize_matrix(width = res, height = res, method = 'bilinear')
    } 
    temp_x[i,,] <- candidate_x
    temp_y1[i,,] <- candidate_y1
    temp_y2[i,,] <- candidate_y2
  }
  output[['x']] <- temp_x
  output[['y1']] <- temp_y1
  output[['y2']] <- temp_y2
  return(output)
}

## run dice() function through entire set of 26 images
set.seed(777)
dice_list <- list()
pb <- txtProgressBar(min = 0, max = images, initial = 0, style = 3)
for (pic in 1:images) {
  dice_list[[pic]] <- dice(
    image_x = x[[pic]],
    image_y1 = y_lung[[pic]],
    image_y2 = y_air[[pic]],
    sample = sample,
    res = res
  )
  setTxtProgressBar(pb, pic)
}; rm(pb, pic)

## remove old x and y formats to free up memory
rm(x, y_lung, y_air)

## convert dice list to x and y arrays
x <- array(dim = c(length(dice_list)*sample, res, res))
y <- array(dim = c(length(dice_list)*sample, res, res, 2)) # last dimension of y is binary class annotation
pb <- txtProgressBar(min = 0, max = images, initial = 0, style = 3)
for (i in 1:length(dice_list)) {
  x[(((i-1)*sample)+1):(i*sample),,] <- dice_list[[i]][['x']]
  y[(((i-1)*sample)+1):(i*sample),,,1] <- dice_list[[i]][['y1']]
  y[(((i-1)*sample)+1):(i*sample),,,2] <- dice_list[[i]][['y2']]
  setTxtProgressBar(pb, i)
  rm(i)
}; rm(pb)

## remove list of images to save memory (frees up 20 GB)
rm(dice_list)

## convert [y > 0] <- 1 (to binary classification; current y ranges from 0 -> 0.3)
y[y > 0] <- 1

## shuffle order of images
set.seed(777)
shuffle <- sample(1:dim(x)[1])
x <- x[shuffle,,]
y <- y[shuffle,,,]
rm(shuffle)

## split into training and test sets
set.seed(777)
split <- createDataPartition(1:dim(x)[1], p = 0.2) |> unlist()
x_train <- x[-c(split),,]
y_train <- y[-c(split),,,]
x_test <- x[split,,]
y_test <- y[split,,,]
rm(split)

## remove old arrays to clear memory (clears 20 GB)
rm(x, y)

#### U-Net Architecture #### 
model <- imageseg::u_net(
  res,
  res,
  grayscale = TRUE,
  blocks = 2,
  n_class = 1
)
summary(model)
model |> deepviz::plot_model()

# #### custom model based on Unet (this doesn't perform very well without the residual connections, but I didn't want to go throught the trouble of writing my own code for that)
# inputs <- layer_input(shape = c(dim(x_train)[-1], 1))
# outputs <- inputs |> 
#   layer_conv_2d(filters = 16, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |> 
#   layer_conv_2d(filters = 16, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_max_pooling_2d(pool_size = c(2,2), padding = 'same') |> 
#   
#   layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_max_pooling_2d(pool_size = c(2,2), padding = 'same') |> 
#   
#   layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_max_pooling_2d(pool_size = c(2,2), padding = 'same') |> 
#   
#   layer_conv_2d(filters = 128, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d(filters = 128, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_max_pooling_2d(pool_size = c(2,2), padding = 'same') |> 
#   
#   layer_conv_2d(filters = 256, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d(filters = 256, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_max_pooling_2d(pool_size = c(2,2), padding = 'same') |>
#   
#   layer_upsampling_2d(size = c(2,2)) |> 
#   layer_conv_2d_transpose(filters = 256, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d_transpose(filters = 256, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |> 
#   
#   layer_upsampling_2d(size = c(2,2)) |> 
#   layer_conv_2d_transpose(filters = 128, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d_transpose(filters = 128, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |> 
#   
#   layer_upsampling_2d(size = c(2,2)) |> 
#   layer_conv_2d_transpose(filters = 64, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d_transpose(filters = 64, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |> 
#   
#   layer_upsampling_2d(size = c(2,2)) |> 
#   layer_conv_2d_transpose(filters = 16, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |>
#   layer_conv_2d_transpose(filters = 16, kernel_size = c(3,3), padding = 'same') |> 
#   layer_batch_normalization() |> 
#   layer_activation_relu() |> 
#   
#   layer_upsampling_2d(size = c(2,2)) |> 
#   layer_conv_2d_transpose(filters = 1, kernel_size = c(3,3), padding = 'same') |> 
#   layer_dropout(0.5) |> 
#   layer_activation_softmax()
# model <- keras_model(inputs, outputs)
# model

## compile
model |> compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = loss_dice(),
  metrics = metric_dice_coeff()
)

## Need to train separate model for lungs and airway, since classes aren't mutually exclusive (which Dice loss can't handle without modification)
## for y_train: 1 = lung, 2 = airways
model_lung <- model
history_lung <- model_lung |> fit(
  x = x_train,
  y = y_train[,,,1],
  batch_size = 32,
  epochs = epochs,
  validation_split = 0.3,
  # callbacks = list(
  #   callback_early_stopping(
  #     monitor = "val_loss",
  #     patience = 5,
  #     restore_best_weights = TRUE))
)
setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/hw3 final models')
save_model_weights_hdf5(model_lung, 'lung256.hdf5')

## now train airway model
model_air <- model
history_air <- model_air |> fit(
  x = x_train,
  y = y_train[,,,2],
  batch_size = 32,
  epochs = epochs,
  validation_split = 0.3,
  # callbacks = list(
  #   callback_early_stopping(
  #     monitor = "val_loss",
  #     patience = 5,
  #     restore_best_weights = TRUE))
)
setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/hw3 final models')
save_model_weights_hdf5(model_air, 'airway256.hdf5')

#### evaluate model on training set ####
## for whatever reason, keras or tensorflow won't let me have 2 models loaded at the same time, even if they have different names
trained <- function(class = c('lung', 'airway'), model_base) {
  setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/hw3 final models')
  load_model_weights_hdf5(model_base, paste0(class, '256.hdf5'))
}
evaluate(trained('lung', model), x_test, y_test[,,,1]) # Lung accuracy
evaluate(trained('airway', model), x_test, y_test[,,,2]) # Airway accuracy

## plot model predictions
pred <- predict(trained('lung', model), x_test)
x_test[50,,] |> as.raster() |> plot()
y_test[50,,,1] |> as.raster() |> plot()
pred[50,,,1] |> as.raster() |> plot()

pred <- predict(trained('airway', model), x_test)
y_test[5,,,2] |> as.raster() |> plot()
pred[5,,,] |> as.raster() |> plot()


###########################
#### import the 27th image, and create a .gif to compare true vs. predicted segmentation patterns
library(RNifti) # readNifti(), channels()
greyscale <- function(RGB) {
  if (length(dim(RGB)) > 2) {
    RGB[,,,1]*0.3 + RGB[,,,2]*0.59 + RGB[,,,3]*0.11
  } else {
    RGB
  }
}

x_27 <- list()
y_lung_27 <- list()
y_air_27 <- list()
setwd(paste0('C:/Users/Matt/Desktop/STAT 590B/hw3/aeropath/', 27))
image <- readNifti(paste0(27, '_CT_HR.nii.gz')) |> channels() |> greyscale()
x_27 <- image/255
setwd(paste0('C:/Users/Matt/Desktop/STAT 590B/hw3/aeropath/', 27))
image <- readNifti(paste0(27, '_CT_HR_label_lungs.nii.gz')) |> channels() |> greyscale()
y_lung_27 <- image
setwd(paste0('C:/Users/Matt/Desktop/STAT 590B/hw3/aeropath/', 27))
image <- readNifti(paste0(27, '_CT_HR_label_airways.nii.gz')) |> channels() |> greyscale()
y_air_27 <- image

## convert lists of images to an array, and resize to 128x128
x_27_input <- array(dim = c(723, res, res, 1))
y_lung_27_input <- array(dim = c(723, res, res, 1))
y_air_27_input <- array(dim = c(723, res, res, 1))
pb <- txtProgressBar(min = 0, max = 723, style = 3)
for (i in 1:723) {
  x_27_input[i,,,] <- x_27[,,i] |> resize_matrix(width = res, height = res, method = 'bilinear')
  y_lung_27_input[i,,,] <- y_lung_27[,,i] |> resize_matrix(width = res, height = res, method = 'bilinear')
  y_air_27_input[i,,,] <- y_air_27[,,i] |> resize_matrix(width = res, height = res, method = 'bilinear')
  
  setTxtProgressBar(pb, i)
  rm(i)
}

## convert y data to binary classification
y_lung_27_input[y_lung_27_input > 0] <- 1
y_air_27_input[y_air_27_input > 0] <- 1

## remove last channel of each input so the number of channels batches trained model
dim(y_air_27_input) <- dim(y_lung_27_input) <- dim(x_27_input) <- c(dim(x_27_input)[1:3])

## use the resized inputs from image 27 to predict masks
evaluate(trained('lung', model), x_27_input, y_lung_27_input, batch_size = 32)
evaluate(trained('airway', model), x_27_input, y_air_27_input, batch_size = 32)
pred_lung <- predict(trained('lung', model), x_27_input)
pred_air <- predict(trained('airway', model), x_27_input)
dim(pred_lung) <- dim(pred_lung)[1:3]
dim(pred_air) <- dim(pred_air)[1:3]
x_27_input[500,,] |> as.raster() |> plot()
y_lung_27_input[500,,] |> as.raster() |> plot()
pred_lung[500,,] |> as.raster() |> plot()

pred_lung_thresh <- pred_lung
pred_lung_thresh[pred_lung_thresh > 0.1] <- 1
pred_air_thresh <- pred_air
pred_air_thresh[pred_air_thresh > 0.1] <- 1

  
  
for (i in 1:723) {
  setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/hw3 final models/27')
  temp <- cbind(x_27_input[i,,], y_lung_27_input[i,,], pred_lung_thresh[i,,])
  temp2 <- cbind(x_27_input[i,,], y_air_27_input[i,,], pred_air_thresh[i,,])
  plot <- rbind(temp, temp2)
  png::writePNG(plot, paste0(i,'.png'))
  rm(i, temp, temp2, plot)
}

## import the pngs you just saved and create gif
library(magick)
image_list <- list()
for (i in 1:723) {
  image_list[[i]] <- image_read(paste0(i, '.png'))
  rm(i)
}

## animate list of images
image_gif <- image_list |> image_join() |> image_animate(fps = 50)
setwd('C:/Users/Matt/Desktop/STAT 590B/hw3/hw3 final models')
image_write(
  image = image_gif,
  path = '256_27_threshold.gif'
)
