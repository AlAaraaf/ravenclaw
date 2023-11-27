








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
slices <- sample(1:depth, size = n_sample_each, replace = FALSE)  # make replace = TRUE if n_sample_each > min(depth) = 265

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
