# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
import os
import info

def predict(modelName):
    model = load_model(info.modelDir + modelName)

    if ( model == None):
        print("No model to load")
        return

    predictDir = info.dataDir + "predict/"
    for file in os.listdir(predictDir):
        if (not os.path.isfile(predictDir)):
            continue
        
        img = load_image(predictDir + file)

        result = model.predict(img)

        print("File: " + file)
        print("Result: " + str(result))

# load and prepare the image
def load_image(filename,arguments):
    if (arguments.model == "vgg16"):
        targetSize = 224
    else: targetSize = 200
    
	# load the image
    img = load_img(filename, target_size=(targetSize, targetSize))
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 3 channels
    img = img.reshape(1, targetSize, targetSize, 3)
	# center pixel data
    img = img.astype('float32')
	#img = img - [123.68, 116.779, 103.939]
    return img