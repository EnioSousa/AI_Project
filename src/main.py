from dataProcess import *
from model import *
import tensorflow as tf
import argparse
import predict

def generateModels(arguments: argparse.ArgumentParser):
    run_test_harness(arguments)

def gpuMemorygrowth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="The model name to use in keras")
    parser.add_argument("--imgAgu", action='store_true',
                        help="If present then we use images with noise")
    parser.add_argument("--pandas", action='store_true',
                        help="If present we use the dataset with pandas")
    parser.add_argument("--onlyCreateDir", action='store_true',
                        help="if present, only creates the data directories")
    parser.add_argument("--epoch", type=int,
                        help="Number of epochs")
    parser.add_argument("--predict", action="store_true",
                        help="If present we try to classify a group of images")

    return parser.parse_args()


if __name__ == '__main__':
    parse = parseArguments()

    checkData()
    gpuMemorygrowth()

    if (parse.predict == False and parse.model != None):
        print("Generate models")
        generateModels(parse)

    elif (parse.model != None):
        print("Predict")
        modelName = parse.model

        if ( parse.pandas ):
            modelName += ".pandas"

        if ( parse.imgAgu):
            modelName += ".imgAgu"

        predict.predict(modelName)
