from bg_ai import LogManager
from algorithms.alphazero import AlphaZeroEvaluator
from agents.mcts import MCTSAgent

logger = LogManager().getLogger(__name__)

class AlphaZeroAgent(MCTSAgent):
    def initAgent(self, model, provider, checkpointFile="alphazero_best", **kwargs):
        provider1 = provider(model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))
        provider1.initModel()
        if checkpointFile:
            if not provider1.load_checkpoint(repr(self.game), checkpointFile):
                raise Exception("Unable to find checkpoint")
            
        self.kwargs=kwargs

        super().initAgent(evaluator=AlphaZeroEvaluator(provider1), **self.kwargs)



