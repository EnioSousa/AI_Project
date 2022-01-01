from matplotlib import pyplot
from matplotlib.image import imread
from keras.preprocessing.image import img_to_array, array_to_img, load_img, save_img
import os
import sys
import shutil
import random as rnd
import numpy as np
import getPandasImages
import logging
import info


def testDogPlot():
    """ Plots none images from dogs dataset
    """
    logging.info("Ploting dogs images")

    folder = 'data/train/'

    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        filename = folder + 'dog.' + str(i) + '.jpg'
        image = imread(filename)
        pyplot.imshow(image)

    pyplot.show()


def testCatPlot():
    """ Plot nine images from cats dataset
    """
    logging.info("Ploting cats images")

    folder = 'data/train/'

    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        filename = folder + 'cat.' + str(i) + '.jpg'
        image = imread(filename)
        pyplot.imshow(image)

    pyplot.show()


def checkData():
    """ checkData will check if we have the correct data folders and files. If the
    folders are non existent then it will create the folders. If the files
    from the panda data set are non existent, then it will download them from
    the internet. 
    """

    folderCreation()

    # check if we have the default dataset
    if (not os.listdir(info.dataDir + "train/")):
        logging.warning("No default dataset")

        try:
            dataSelection()
        except Exception as e:
            logging.exception("Exception while performing data selection")
            sys.exit(1)

    # check if we have the default noise data set
    if (not os.listdir(info.dataDir + "trainNoise/" + "cats/")):
        logging.warning("No images for default dataset with noise")
        createWhiteNoise(info.dataDir + "train/cats/",
                         info.dataDir + "trainNoise/cats/")
        createWhiteNoise(info.dataDir + "train/dogs/",
                         info.dataDir + "trainNoise/dogs/")

    # Check if we have the panda data set
    if (not os.listdir(info.dataDir + "trainPanda/" + "panda/")):
        logging.warning("No panda dataset")

        createFolder(info.dataDir + "rawPanda")
        getImages("panda", info.dataDir + "rawPanda")
        getImages("giant panda", info.dataDir + "rawPanda")
        getImages("panda animal", info.dataDir + "rawPanda")

        try:
            dataSelection(info.dataDir + "rawPanda")
        except Exception as e:
            logging.exception("Exception while performing data selection")
            sys.exit(1)

    # check if we have the panda noise data set
    if (not os.listdir(info.dataDir + "trainPandaNoise/" + "panda/")):
        logging.warning("No images for panda dataset with noise")
        createWhiteNoise(info.dataDir + "trainPanda/panda/",
                         info.dataDir + "trainPandaNoise/panda/")

    createFolder(info.modelDir)
    createFolder(info.plotDir)


def getImages(query: str, dst: str):
    """ Get images from bing

    Args:
        query (str): query to use
        dst (str): destination folder
    """

    logging.info("Get images for %s", query)

    listDir = os.listdir(dst)
    getPandasImages.getPandaImages(query, dst, len(listDir))


def folderCreation():
    """ _folderCreation will create the appropriate folders for the panda and default
    data sets. For the panda dataset it will create symbolic links to the dogs and 
    cats data set 
    """

    logging.info("Creating dataset folders")

    # create directories
    datasetHome = info.dataDir
    subDirs = ['train/', 'test/', 'trainNoise/']

    for subDir in subDirs:
        createFolder(datasetHome + subDir)

        # create label subdirectories
        for endDir in ['dogs/', 'cats/']:
            newDir = datasetHome + subDir + endDir
            createFolder(newDir)

    createFolder(datasetHome + "trainPanda/panda")
    createSymbLink("../train/cats/", datasetHome + "trainPanda/cats")
    createSymbLink("../train/dogs/", datasetHome + "trainPanda/dogs")

    createFolder(datasetHome + "trainPandaNoise/panda")
    createSymbLink("../trainNoise/cats/", datasetHome + "trainPandaNoise/cats")
    createSymbLink("../trainNoise/dogs/", datasetHome + "trainPandaNoise/dogs")

    createFolder(datasetHome + "testPanda/panda")
    createSymbLink("../test/cats/", datasetHome + "testPanda/cats")
    createSymbLink("../test/dogs/", datasetHome + "testPanda/dogs")


def createFolder(path: str):
    """
    Create folders and logs the execution result

    Args:
        path (str): Path to directory
    """
    if (not os.path.exists(path)):
        os.makedirs(path)
        logging.info("Created %s", path)
    else:
        logging.info("Exists %s", path)


def createSymbLink(src: str, dst: str):
    """Create symbolic link

    Args:
        src (str): relative path from dst
        dst (str): where to put the link
    """
    if (not os.path.exists(dst)):
        os.system("ln -sf " + src + " " + dst)
        logging.info("Symb link created %s -> %s", dst, src)
    else:
        logging.info("Symb link exists %s", dst)


def dataSelection(dataSrc: str):
    """ Creates a test and train dataset from a source file

    Args:
        dataSrc (str): path to the source file
    """

    logging.info("Creating default dataset")
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

    shutil.rmtree(dataSrc)


def createWhiteNoise(sourceDir: str, destDir: str):
    """Creates images with white noise for a given source directory containing 
    said images and puts them in a new directory. The original images are also 
    copied

    Args:
        sourceDir (str): Source directory containing the images
        destDir (str): Destination directory to put the images
    """

    logging.info("Creating white noise images for %s", sourceDir)

    try:
        for file in os.listdir(sourceDir):
            if (file.endswith(".jpg")):
                shutil.copyfile(sourceDir + file, destDir + file)
                image = load_img(sourceDir + file)
                save_img(destDir + "noise." + file, getNoisyImage(image))
    except Exception as e:
        logging.exception("Exception while creating white noise")
        sys.exit(1)


def getNoisyImage(image):
    """ For a given imagem, adds white noise

    Args:
        image (PIL image): PIL image

    Returns:
        PIL image: PIL image with noise
    """

    imageArray = img_to_array(image)
    row, col, ch = imageArray.shape
    mean = 0.9
    var = 0.6
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + (gauss * 100)
    return array_to_img(noisy)
