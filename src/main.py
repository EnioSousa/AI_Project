from dataProcess import *
from model import *
import tensorflow as tf
import argparse
import predict
import logging
import model

def gpuMemorygrowth():
    """Activates gpu memory growth
    """
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


def parseArguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="The model name to use in keras")
    parser.add_argument("--imgAgu", action='store_true',
                        help="If present then we use images with noise")
    parser.add_argument("--pandas", action='store_true',
                        help="If present we use the dataset with pandas")
    parser.add_argument("--checkData", action='store_true',
                        help="if present, only creates the data directories")
    parser.add_argument("--generate", action="store_true",
                        help="If present will try to generate the network")
    parser.add_argument("--epoch", type=int,
                        help="Number of epochs")
    parser.add_argument("--predict", action="store_true",
                        help="If present we try to classify a group of images")
    parser.add_argument("--log", action="store_true",
                        help="If present we put the logs under the model name, otherwise std.log")
    parser.add_argument("--gauss", action="store_true",
                        help="If present we use images with white noise")
    parser.add_argument("--featureDesc", action="store_true",
                        help="If present we try to use the image descriptions")

    return parser


if __name__ == '__main__':
    parser = parseArguments()
    args = parser.parse_args()

    if (args.generate):
        if (args.epoch == None and (not args.featureDesc)):
            parser.error("--epoch or --featureDesc is required when --generate is set")
            sys.exit(1)

        if (args.model == None and (not args.featureDesc)):
            parser.error("--model or --featureDesc is required when --generate is set")
            sys.exit(1)

    if (args.predict):
        if (args.model == None):
            parser.error("--model is required when --predict is set")
            sys.exit(1)

    modelName = None

    if (args.model != None):
        modelName = args.model
       
        if (args.pandas):
            modelName += ".pandas"

        if (args.imgAgu):
            modelName += ".imgAgu"

        if ( args.gauss):
            modelName += ".gauss"

    if (args.log and args.model != None):
        logging.basicConfig(filename=info.logDir + modelName + ".log",
                            level=logging.INFO, filemode='w')
    else:
        logging.basicConfig(filename=info.logDir + "std.log",
                            filemode='w', level=logging.INFO,)

    logging.info("start")

    checkData()
    gpuMemorygrowth()

    if (args.generate):
        try:
            model.trainModel(modelName, args)
        except Exception as e:
            logging.exception("Exception while trying to train")

    if (args.predict):
        try:
            predict.predict(modelName)
        except Exception as e:
            logging.exception("Exception while trying to predict")
