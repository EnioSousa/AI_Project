import os
import sys
import logging
from cv2 import destroyAllWindows
from keras.saving.save import load_model

from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import info
import argparse
import modelResnet
import plot
import modelFeaExt


def modelVGG1(nClass: int) -> Sequential:
    """ Creates a vgg1 model

    Args:
        nClass (int): Number of classes in our dataset

    Returns:
        Sequential: vgg model
    """
    logging.info("Model vgg1 chosen")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nClass, activation="softmax"))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


def modelVGG2(nClass: int) -> Sequential:
    """ Creates a vgg2 model

    Args:
        nClass (int): Number of classes in our dataset

    Returns:
        Sequential: vgg model
    """
    logging.info("Model vgg2 chosen")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nClass, activation="softmax"))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


def modelVGG3(nClass: int) -> Sequential:
    """ Creates a vgg3 model

    Args:
        nClass (int): Number of classes in our dataset

    Returns:
        Sequential: vgg model
    """
    logging.info("Model vgg3 chosen")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nClass, activation="softmax"))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


# define cnn model
def modelVGG16(nClass: int):
    logging.info("Model vgg16 chosen")

    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu',
                   kernel_initializer='he_uniform')(flat1)
    output = Dense(nClass, activation='softmax')(class1)

    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


def modelDropout(nClass: int):
    """ create a vgg model with dropout

    Args:
        nClass (int): number of classes in the dataset

    Returns:
        [type]: vgg model with dropout
    """
    logging.info("Model dropout chosen")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation="softmax"))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def trainModel(modelName: str, args: argparse.ArgumentParser.parse_args):
    logging.info("Starting to train model")

    # Calculate the number of classes in dataset
    nClass = 2
    if (args.pandas):
        nClass += 1


    if (args.featureDesc):
        modelFeaExt.linearModel(args)
        return

    logging.info("Choosing model type")

    # Specify data set source
    testSource = info.dataDir
    trainSource = info.dataDir

    if (args.gauss):
        if (args.pandas):
            trainSource += "trainPandaNoise/"
            testSource += "testPanda"

        else:
            trainSource += "trainNoise/"
            testSource += "test/"

    else:
        if (args.pandas):
            trainSource += "trainPanda/"
            testSource += "testPanda"

        else:
            trainSource += "train/"
            testSource += "test/"

    # Chose model
    if (args.model == "vgg1"):
        model = modelVGG1(nClass)

    elif(args.model == "vgg2"):
        model = modelVGG2(nClass)

    elif(args.model == "vgg3"):
        model = modelVGG3(nClass)

    elif(args.model == "vgg16"):
        model = modelVGG16(nClass)

    elif (args.model == "dropout"):
        model = modelDropout(nClass)

    elif(args.model == "resNet50"):
        try:
            modelResnet.trainModelResNet50(nClass, modelName, trainSource,
                                           testSource)
        except Exception as e:
            logging.exception("Exception while doing resNet50")
        return

    logging.info("Choosing image generator type")
    # Chose data generator
    if ("vgg16" in modelName):
        dataGen = ImageDataGenerator(featurewise_center=True,
                                     horizontal_flip=args.imgAgu)
        dataGen.mean = [123.68, 116.779, 103.939]
        
    elif ("imgAgu" in modelName):
        dataGen = ImageDataGenerator(rescale=1.0/255.0,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True)

    else:
        dataGen = ImageDataGenerator(rescale=1.0/255.0)

    # Specify images target size
    if (args.model == "vgg16"):
        targetSize = (224, 224)
    else:
        targetSize = (200, 200)

    logging.info("Setting data generators")
    trainGen = dataGen.flow_from_directory(trainSource,
                                           class_mode="categorical",
                                           batch_size=info.batchNumber,
                                           target_size=targetSize)

    testGen = dataGen.flow_from_directory(testSource,
                                          class_mode="categorical",
                                          batch_size=info.batchNumber,
                                          target_size=targetSize)

    logging.info("Traning model")
    # history = model.fit(trainGen,
    #                     steps_per_epoch=len(trainGen),
    #                     validation_data=testGen,
    #                     validation_steps=len(testGen),
    #                     epochs=args.epoch)

    history = model.fit(trainGen,
                        steps_per_epoch=1,
                        validation_data=testGen,
                        validation_steps=1,
                        epochs=args.epoch)

    
    logging.info("Traning finished")
    (loss, acc) = model.evaluate(testGen, steps=len(testGen))
    print('> %.3f' % (acc * 100.0))
    logging.info("Accuracy is %.3f", (acc * 100.0))

    plot.plotGraph(history, modelName)

    saveModel(model, modelName)


def saveModel(model: Sequential, modelName: str):
    """Save sequential model

    Args:
        model (Sequential): model to save
        modelName (str): name to be used to save the model
    """
    logging.info("saving model %s", modelName)

    model.save(info.modelDir + modelName)
