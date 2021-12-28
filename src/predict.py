# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import info

def predict(modelName):
    model = load_model(info.modelDir + modelName)

    if ( model == None):
        print("No model to load")
        return

    for file in os.listdir(info.dataDir + "predict"):
        if (not os.path.isfile(file)):
            continue
        
        img = load_image(file)

        result = model.predict(img)

        print("File: " + file)
        print("Result: " + result)

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
