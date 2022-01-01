import info
import logging
from matplotlib import pyplot


def plotGraph(history, modelName: str):
    """ Plot the accuracy and loss of our model traning. The plots are
    saved in the plot directory, with the model name

    Args:
        history ([type]): traning history
        modelName (str): model name
    """
    logging.info("Plot")

    figure, (ax1, ax2) = pyplot.subplots(nrows=2)

    ax1.set_title('Cross Entropy loss')
    ax1.plot(history.history['loss'], color='blue', label='train')
    ax1.plot(history.history['val_loss'], color='orange', label='test')

    ax2.set_title('Classification Accuracy')
    ax2.plot(history.history['accuracy'], color='blue', label='train')
    ax2.plot(history.history['val_accuracy'], color='orange', label='test')

    figure.tight_layout(pad=2)

    try:
        figure.savefig(info.plotDir + modelName + ".png")
    except Exception as e:
        logging.exception("Exception while saving plot")

    logging.info("Plot saved")
    pyplot.close()
