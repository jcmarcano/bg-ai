from bg_ai import Provider
import numpy as np
import os

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

    def save_checkpoint(self, gameName, checkPointName, iteration=None):
        filename = self.getFileName(gameName, checkPointName, iteration)
        
        filepath = os.path.join(self.folder, filename)
        self.kerasModel.save_weights(filepath)

    def load_checkpoint(self, gameName, checkPointName, iteration=None):
        # change extension
        filename = self.getFileName(gameName, checkPointName, iteration)
        filepath = os.path.join(self.folder, filename)
        if not os.path.isfile(filepath):
            return False
        
        self.kerasModel.load_weights(filepath)
        return True
