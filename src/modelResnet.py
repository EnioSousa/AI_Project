from enum import Flag
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# from keras.preprocessing import image
import info
import logging
import plot


def modelResNet50(nClass: int):
    # Download the architecture of ResNet50 with ImageNet weights
    base_model = ResNet50(include_top=False, weights='imagenet',
                          input_shape=(224, 224, 3))

    # We won't train the input layers
    for layer in base_model.layers:
        layer.trainable = False

    # Taking the output of the last convolution block in ResNet50
    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)

    # Adding a fully connected layer having 2/3 neurons which will
    # give the probability of image having either dog or cat or panda
    predictions = Dense(nClass, activation='softmax')(x)

    # Model to be trained
    model = Model(inputs=base_model.input, outputs=predictions)

    # IF the modelruns like shit, lets try another optimizer
    opt = SGD(lr=0.0001, momentum=0.9)

    # Compiling the model
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def trainModelResNet50(nClass: int, modelName: str, trainDir: str, testDir: str):
    model = modelResNet50(nClass)

    # Creating objects for image augmentations
    logging.info("Choosing image generator type")
    trainDataGen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)

    testDataGen = ImageDataGenerator(rescale=1./255)

    logging.info("Setting data generators")
    trainGen = trainDataGen.flow_from_directory(trainDir,
                                                target_size=(224, 224),
                                                class_mode='categorical',
                                                batch_size=info.batchNumber)

    testGen = testDataGen.flow_from_directory(testDir,
                                              target_size=(224, 224),
                                              batch_size=info.batchNumber,
                                              class_mode='categorical')

    # Training the model for 10 epochs
    logging.info("Traning for 10 epochs")
    history = model.fit_generator(trainGen,
                                  steps_per_epoch=len(trainGen),
                                  epochs=10,
                                  validation_data=testGen,
                                  validation_steps=len(testGen))

    logging.info("Traning finished")

    (loss, acc) = model.evaluate(testGen, steps=len(testGen))
    print('> %.3f' % (acc * 100.0))
    logging.info("Accuracy is %.3f", (acc * 100.0))

    plot.plotGraph(history, modelName)

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
