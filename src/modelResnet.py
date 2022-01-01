from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

import info

# TODO: Fix the direcotries paths
# TODO: Change this file to accept panda directory

def modelResNet50(nClass: int):
    # Download the architecture of ResNet50 with ImageNet weights
    base_model = ResNet50(include_top=False, weights='imagenet')

    # Taking the output of the last convolution block in ResNet50
    x = base_model.output

    # Adding a Global Average Pooling layer
    x = GlobalAveragePooling2D()(x)

    # Adding a fully connected layer having 1024 neurons
    x = Dense(1024, activation='relu')(x)

    # Adding a fully connected layer having 2 neurons which will
    # give the probability of image having either dog or cat
    predictions = Dense(nClass, activation='softmax')(x)

    # Model to be trained
    model = Model(inputs=base_model.input, outputs=predictions)

    # Training only top layers i.e. the layers which we have added in the end
    for layer in base_model.layers:
        layer.trainable = False
    # Compiling the model
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return base_model, model


def trainModelResNet50():
    baseModel, model = modelResNet50(2)  # Change

    # Creating objects for image augmentations
    trainDataGen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)

    testDataGen = ImageDataGenerator(rescale=1./255)

    # Proving the path of training and test dataset
    # Setting the image input size as (224, 224)
    # We are using class mode as binary because there are only two classes in our data
    trainSet = trainDataGen.flow_from_directory('../data/train',
                                                target_size=(224, 224),
                                                class_mode='categorical')

    testSet = testDataGen.flow_from_directory('../data/test',
                                              target_size=(224, 224),
                                              batch_size=64,
                                              class_mode='categorical')

    # Training the model for 5 epochs
    model.fit_generator(trainSet,
                        steps_per_epoch=len(trainSet),
                        epochs=5,
                        validation_data=testSet,
                        validation_steps=len(testSet))

    # We will try to train the last stage of ResNet50
    for layer in baseModel.layers[0:143]:
        layer.trainable = False

    for layer in baseModel.layers[143:]:
        layer.trainable = True

    # Training the model for 10 epochs
    model.fit_generator(trainSet,
                        steps_per_epoch=len(trainSet),
                        epochs=10,
                        validation_data=testSet,
                        validation_steps=len(testSet))

    # Saving the weights in the current directory
    model.save_weights(info.modelDir + "resnet50_weights.h5")

# def predict():

#     # Predicting the final result of image
#     test_image = image.load_img('../data/predict/cat.0.jpg', target_size = (224, 224))
#     test_image = image.img_to_array(test_image)\

#     # Expanding the 3-d image to 4-d image.
#     # The dimensions will be Batch, Height, Width, Channel
#     test_image = np.expand_dims(test_image, axis = 0)

#     # Predicting the final class
#     result = model.predict(test_image)[0].argmax()

#     # Fetching the class labels
#     labels = training_set.class_indices
#     labels = list(labels.items())

#     # Printing the final label
#     for label, i in labels:
#         if i == result:
#             print("The test image has: ", label)
#             break
