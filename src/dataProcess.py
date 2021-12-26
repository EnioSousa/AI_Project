# plot dog photos from the dogs vs cats dataset
from genericpath import exists
import shutil
from matplotlib import pyplot
from matplotlib.image import imread
import os
from pathlib import Path
from shutil import copyfile
import random as rnd
import info
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img, load_img, save_img
import matplotlib.pyplot as plt
import getPandasImages


def testDogPlot():
    # define location of dataset
    folder = 'data/train/'
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'dog.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()


def testCatPlot():
    folder = 'data/train/'
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'cat.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()


def checkData():
    # Create the basic data folders
    folderCreation()

    # check if we have the default dataset
    if (not os.listdir(info.dataDir + "train/")):
        print("No default dataset")
        dataSelection()

    # check if we have the default noise data set
    if (not os.listdir(info.dataDir + "trainNoise/" + "cats/")):
        print("trainNoise directory doesn't cointain images")
        createWhiteNoise(info.dataDir + "train/cats/",
                         info.dataDir + "trainNoise/cats/")
        createWhiteNoise(info.dataDir + "train/dogs/",
                         info.dataDir + "trainNoise/dogs/")

    # Check if we have the panda data set
    if (not os.listdir(info.dataDir + "trainPanda/" + "panda/")):
        print("No images of panda")
        os.makedirs(info.dataDir + "rawPanda", exist_ok=True)
        getPandasImages.getPandaImages("panda",
                                       info.dataDir + "rawPanda", 0)
        getPandasImages.getPandaImages("Giant panda",
                                       info.dataDir + "rawPanda", 1000)
        getPandasImages.getPandaImages("panda animal",
                                       info.dataDir + "rawPanda", 2000)
        dataSelection(info.dataDir + "rawPanda")
        shutil.rmtree(info.dataDir + "rawPanda")

    # check if we have the panda noise data set
    if (not os.listdir(info.dataDir + "trainPandaNoise/" + "panda/")):
        print("No noise for panda set")
        createWhiteNoise(info.dataDir + "trainPanda/panda/",
                         info.dataDir + "trainPandaNoise/panda/")

    # Create model, plot directies. They will be used to store the
    # model and plots
    os.makedirs(info.modelDir, exist_ok=True)
    os.makedirs(info.plotDir, exist_ok=True)


def folderCreation():
    print("Creating folders")

    # create directories
    dataset_home = info.dataDir
    subdirs = ['train/', 'test/', 'trainNoise/']

    for subdir in subdirs:
        os.makedirs(dataset_home + subdir, exist_ok=True)

        # create label subdirectories
        for labldir in ['dogs/', 'cats/']:
            newdir = dataset_home + subdir + labldir
            os.makedirs(newdir, exist_ok=True)

    # Make directories for the panda data set
    os.makedirs(dataset_home + "trainPanda/", exist_ok=True)
    os.makedirs(dataset_home + "trainPanda/panda", exist_ok=True)
    os.system("ln -sf " + "../train/cats/ " + dataset_home + "trainPanda/cats")
    os.system("ln -sf " + "../train/dogs/ " + dataset_home + "trainPanda/dogs")

    os.makedirs(dataset_home + "trainPandaNoise/", exist_ok=True)
    os.makedirs(dataset_home + "trainPandaNoise/panda", exist_ok=True)
    os.system("ln -sf " + "../trainNoise/cats/ " +
              dataset_home + "trainPandaNoise/cats")
    os.system("ln -sf " + "../trainNoise/dogs/ " +
              dataset_home + "trainPandaNoise/dogs")

    os.makedirs(dataset_home + "testPanda/", exist_ok=True)
    os.makedirs(dataset_home + "testPanda/panda", exist_ok=True)
    os.system("ln -sf " + "../test/cats/ " + dataset_home + "testPanda/cats")
    os.system("ln -sf " + "../test/dogs/ " + dataset_home + "testPanda/dogs")

def dataSelection(dataSrc: str):
    print("Creating dataset")
    rnd.seed(1)

    # define ratio of pictures to use for validation
    val_ratio = 0.25

    for file in os.listdir(dataSrc):
        src = dataSrc + '/' + file

        if (src.endswith(".jpg")):
            if rnd.random() < val_ratio:
                dst_dir = 'test'
            else:
                dst_dir = 'train'

            if file.startswith('cat'):
                dst = info.dataDir + dst_dir + "/" + 'cats/' + file
                os.rename(src, dst)

            elif file.startswith('dog'):
                dst = info.dataDir + dst_dir + "/" + 'dogs/' + file
                os.rename(src, dst)

            elif file.startswith('panda'):
                dst = info.dataDir + dst_dir + "Panda/" + 'panda/' + file
                os.rename(src, dst)


def createWhiteNoise(sourceDir: str, destDir: str):
    print("Creating white noise images")

    for file in os.listdir(sourceDir):
        if (file.endswith(".jpg")):
            copyfile(sourceDir + file, destDir + file)
            image = load_img(sourceDir + file)
            save_img(destDir + "noise." + file, getNoisyImage(image))

def getNoisyImage(image):
    imageArray = img_to_array(image)
    row, col, ch = imageArray.shape
    mean = 0.9
    var = 0.6
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + (gauss * 100)
    return array_to_img(noisy)
