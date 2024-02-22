from bg_ai import Trainer
from algorithms.alphazero import AlphaZero

class AlphaZeroTrainer(Trainer):
    def initTrainer(self, **kwargs):
        self.alphazero = AlphaZero(self.game, **kwargs)
        
    def train(self, step=None):
        self.alphazero.train(step=step)
