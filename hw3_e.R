####################################
## run for 2(e)
####################################

### Loading the Xception network with pretrained weights
model <- application_xception(weights = "imagenet")
img_path <-"D://iowa state//STAT590s3//hw3//Pomegranate//disease//0020_0001.JPG"
img_tensor <- img_path |> tf_read_image(resize = c(299, 299))
preprocessed_img <- img_tensor[tf$newaxis, , , ] %>% xception_preprocess_input()
preds <- predict(model, preprocessed_img)
str(preds)
imagenet_decode_predictions(preds, top=3)[[1]]
