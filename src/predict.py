# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import Sequential, Model
import os
import sys
import info
import logging

def predict(modelName):
    logging.info("Predict with %s", modelName)

    model = load_model(info.modelDir + modelName)

    if ( model == None):
        logging.error("Model not loaded %s", modelName)
        sys.exit(1)

    else:
        logging.info("Model loaded")

    predictDir = info.dataDir + "predict/"

    for file in os.listdir(predictDir):
        if (not os.path.isfile(predictDir)):
            continue
        
        img = load_image(predictDir + file)

        result = model.predict(img)

        print("File: " + file)
        print("Result: " + str(result))
        
        logging.info("File: %s", file)
        logging.info("Result: %s", str(result))


# load and prepare the image
def load_image(filename:str, args:str):
    logging.info("Loading image %s", filename)
	# load the imagetargetSize
    if ( args.model == "vgg16"):
        img = load_img(filename, target_size=(224, 224))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 3 channels
        img = img.reshape(1, 244, 244, 3)
        # center pixel data
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]

    else:
        img = load_img(filename, target_size=(200, 200))
        # convert to array
        img = img_to_array(img)
        img = img.astype('float32')

    return img