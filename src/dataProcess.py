# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
from os import makedirs
from os import listdir
from shutil import copyfile
import random as rnd

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

def folderCreation():
    # create directories
    dataset_home = 'data/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['dogs/', 'cats/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)

def dataSelection():
    rnd.seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25

    dataset_home = 'data/'

    # copy training dataset images into subdirectories
    src_train_directory = 'Data/train/'

    for file in listdir(src_train_directory):
        src = src_train_directory + '/' + file
        dst_dir = 'train/'
        if rnd.random() < val_ratio:
            dst_dir = 'test/'
        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/'  + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/'  + file
            copyfile(src, dst)
