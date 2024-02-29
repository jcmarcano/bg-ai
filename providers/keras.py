from bg_ai import Provider
import numpy as np
from pathlib import Path

class KerasProvider(Provider):
    def __init__(self, model, inputShapes, outputShapes, batchSize=64, epochs=50, folder='/temp'):
        self.model = model
        self.folder = folder
        self.inputShapes = inputShapes
        self.outputShapes = outputShapes
        self.batchSize = batchSize
        self.epochs = epochs
        self.checkpointExtension = ".h5"

    def initModel(self):
        self.kerasModel = self.model.initModel()

    def train(self, samples):
        """
        samples: list of samples, each sample is of form (inputs, labels)
        """
        sampleInputs, sampleOutputs = list(zip(*samples))

        inputs =  [np.array(input) for input in list(zip(*list(sampleInputs)))]
        outputs = [np.array(output) for output in list(zip(*list(sampleOutputs)))]

        self.kerasModel.fit(x=inputs, y=outputs, batch_size=self.batchSize, epochs=self.epochs)

    def predict(self, input):
        """
        board: np array with board
        """
        inputs = [input[i][np.newaxis,:] for i in range(len(input))]
        outputs = self.kerasModel.predict(x=inputs, verbose=0)

        return tuple(outputs)

    def saveCheckpoint(self, gameName, checkPointName, iteration=None):
        filePath = Path(self.folder) / self.getFileName(gameName, checkPointName, iteration)
        self.kerasModel.save_weights(filePath)

    def loadCheckpoint(self, gameName, checkPointName, iteration=None):
        # change extension
        filePath = Path(self.folder) / self.getFileName(gameName, checkPointName, iteration)
        if not filePath.is_file():
            return False
        
        self.kerasModel.load_weights(filePath)
        return True
