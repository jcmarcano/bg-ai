from bg_ai import State, LogManager, GameSamplesManager, Arena, getDefaultRandomGenerator
from algorithms.mcts import MCTS
from agents.mcts import MCTSAgent
from collections import deque
from p_tqdm import p_map
from tqdm import tqdm
import numpy as np

## AlphaZero implementation based on:
## - Deepmind's OpenSpiel https://github.com/google-deepmind/open_spiel
## - SimpleAlphaZero https://github.com/suragnair/alpha-zero-general

logger = LogManager().getLogger(__name__)

class AlphaZero:
    def __init__(self, game, model, provider, 
                 temperature=1, 
                 temperatureDrop=30, 
                 iterMax=100, 
                 exploration=.7, 
                 maxSampleSize=10000, 
                 selfPlayEpisodes=8, 
                 evalEpisodes=8,
                 parallel=True, 
                 numCPUs=None):
        self.game = game
        self.samples = GameSamplesManager(game, "alphazero")
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.temperatureDrop = temperatureDrop
        self.iterMax=iterMax
        self.exploration=exploration
        self.maxSampleSize = maxSampleSize
        self.selfPlayEpisodes = selfPlayEpisodes
        self.evalEpisodes = evalEpisodes
        self.parallel = parallel
        self.numCPUs = numCPUs

    def executeEpisode(self, episodeNumber, randomGenerator):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training sample to
        trainSamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each sample
        in trainSamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainSamples: a list of samples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        print(f"Starting episode: {episodeNumber}")

        self.game.randomGenerator = randomGenerator

        provider = self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))
        provider.initModel()
        provider.load_checkpoint(repr(self.game), "alphazero_best")

        curPlayer = 0
        state:State = self.game.getInitState()
        agent = MCTSAgent(self.game, iterMax=self.iterMax, exploration=self.exploration, evaluator=AlphaZeroEvaluator(provider))
        
        episodeStep = 0

        observationTensors = []
        while not state.isTerminal():
            episodeStep += 1

            if state.isChanceState():
                setId = state.determinize(randomGenerator)
                agent.externalActionEvent(setId)
            else:
                observation = state.getObservation(curPlayer)
                logger.info(observation)
                print(observation)

                policy = agent.selectPolicy(observation)

                adjustedProbs = np.array([action[1]**(1/self.temperature) for action in policy])
                probSum = sum(adjustedProbs)
                adjustedProbs /= probSum


                # Get observation tensors with symmetries for Player 0
                policyTensors = []

                for i, size in enumerate(self.game.getActionSize()):
                    policyTensors.append(np.zeros(size))
                    for j, actionProb in enumerate(policy):
                        if type(actionProb[0]) is tuple:
                            policyTensors[i][int(actionProb[0][i])] += adjustedProbs[j]
                        else:
                            policyTensors[i][int(actionProb[0])] += adjustedProbs[j]
                observationTensors += observation.getObservationSymmetries(curPlayer, tuple(policyTensors))

                if episodeStep < self.temperatureDrop:
                    selectedActionPos = randomGenerator.choice(len(policy),p=adjustedProbs)
                    action = policy[selectedActionPos][0]
                else:
                    action = max(policy, key=lambda action: action[1])[0]

                print(f"Episode: {episodeNumber}, Step {episodeStep}, Player {curPlayer}, Action {action}")
                logger.info(f"Episode: {episodeNumber}, Step {episodeStep}, Player {state.playerRepr(curPlayer)}, Action {state.actionRepr(action, curPlayer)}")

                state.performAction(action, curPlayer)
                agent.externalActionEvent(action, curPlayer)
                curPlayer = state.getNextPlayer()

        value = state.getResult()[0] # Get Value for player 0

        # Return list of observations tensors for player 0 with policies and final value
        return [(observation[0], observation[1] + (value, )) for observation in observationTensors]
    
    def selfPlay(self, iteration=None):
        if iteration:
            iter = iteration
        else:
            lastIter = self.samples.getLastIteration()
            if lastIter is None:
                iter = 0
            else:
                self.samples.loadTrainSamples(lastIter)
                iter = lastIter + 1

        logger.info(f'Starting Iter #{iter + 1} ...')
        print(f'Starting Iter #{iter + 1} ...')

        if self.parallel:
            for sample in p_map(self.executeEpisode, range(self.selfPlayEpisodes), self.game.randomGenerator.spawn(self.selfPlayEpisodes), num_cpus = self.numCPUs):
                self.samples.addSamples(sample)
        else:
            for episode in tqdm(range(self.selfPlayEpisodes)):
                self.samples.addSamples(self.executeEpisode(episode, self.game.randomGenerator))

        self.samples.saveTrainSamples(iter)

    def trainSamples(self, iteration=None):
        # load trainSamples
        if iteration:
            iter = iteration
        else:
            lastIter = self.samples.getLastIteration()
            if lastIter is None:
                logger.info("There are no samples to train")
                exit(1)
            
            iter = lastIter + 1

        logger.info(f'Train Phase for Iter #{iter} ...')

        provider = self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))
        provider.initModel()

        if not provider.load_checkpoint(repr(self.game), "alphazero_best"):
            provider.save_checkpoint(repr(self.game), "alphazero_best")    

            
        self.samples.loadTrainSamples(lastIter)
        # shuffle samples before training

        # training new network, keeping a copy of the old one

        provider.train(self.samples.trainSamples)
        provider.save_checkpoint(repr(self.game), "alphazero_temp")

    def evaluateGame(self, evalPlayer, provider1, provider2, randomGenerator):
        self.game.randomGenerator = randomGenerator
        provider1.initModel()
        if provider1.load_checkpoint(repr(self.game), "alphazero_temp"):
            agent1 = MCTSAgent(self.game, iterMax=self.iterMax, exploration=self.exploration, evaluator=AlphaZeroEvaluator(provider1))
        else:
            raise Exception("Unable to find 'best' model")

        provider2.initModel()
        if provider2.load_checkpoint(repr(self.game), "alphazero_best"):
            agent2 = MCTSAgent(self.game, iterMax=self.iterMax, exploration=self.exploration, evaluator=AlphaZeroEvaluator(provider2))
        else:
            raise Exception("Unable to find 'temp' model")

        agents = [None, None]
        agents[evalPlayer] = agent1
        agents[1-evalPlayer] = agent2

        # Create Arena
        arena = Arena(self.game, agents, display=True)

        # Play game
        return evalPlayer, arena.playGame()        


    def evaluate(self, iteration=None):
        # Perform episodes evaluation the model using new model as first player and then as second player
        provider1 = self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))
        provider2 = self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))

        totalResult = [0,0]
        evalPlayer = 0

        if self.parallel:
            for evalPlayer, result in p_map(self.evaluateGame, 
                                            [0,1]*(self.evalEpisodes//2) + [0]*(self.evalEpisodes%2), 
                                            [provider1]*self.evalEpisodes, 
                                            [provider2]*self.evalEpisodes, 
                                            self.game.randomGenerator.spawn(self.evalEpisodes),
                                            num_cpus = self.numCPUs):
                totalResult += [result[evalPlayer], result[1-evalPlayer]]
                evalPlayer = 1 - evalPlayer
        else:
            for _ in tqdm(range(self.evalEpisodes)):
                _, result = self.evaluateGame(evalPlayer, provider1, provider2, self.game.randomGenerator)        
                totalResult += [result[evalPlayer], result[1-evalPlayer]]
                evalPlayer = 1 - evalPlayer
        
        #compare models
                
        if totalResult[0] > totalResult[1]:
            # Temp model has better results, accept it as best model
            provider1.save_checkpoint(self, repr(self.game), "alphazero_best")
            provider1.save_checkpoint(self, repr(self.game), iteration=iteration)
            print("Accepting New Model")
        else:
            print("Rejecting Model")


    def train(self, iterations=1, step=None, iteration=None):
        if step is None:
            for i in range(iterations):
                self.selfPlay(i)
                self.trainSamples(i)
                self.evaluate(i)
        elif step=="selfPlay":
            self.selfPlay(iteration)
        elif step=="train":
            self.trainSamples(iteration)
        elif step=="evaluate":
            self.evaluate(iteration)
        else:
            raise Exception(f"Invalid training step: '{step}' (choose from 'selfPlay', 'train', 'evaluate')")



class AlphaZeroEvaluator():
    def __init__(self, provider):
        self.provider = provider

    def prevPolicy(self, state:State):
        # infer action probabilities
        actionProbs = self.provider.predict(state.getObservationTensor(state.getNextPlayer()))[:-1]

        actions = state.getValidActions(state.getNextPlayer())

        policy = []
        sumProb=0
        # Filter valid actions
        for action in actions:
            if type(action) is tuple:
                prob = 1
                for i in range(len(action)):
                    prob *= actionProbs[i][0][int(action[i])]
            else:
                prob = actionProbs[0][0][int(action)]

            policy.append((action, prob))
            sumProb += prob

        return [(action, prob/sumProb) for action,prob in policy ]

    def nodeEvaluator(self, state: State):
        # infer action value
        nextPlayer = state.getNextPlayer()
        playerValue = self.provider.predict(state.getObservationTensor(nextPlayer))[-1][0]
        result = [playerValue[0] if player == nextPlayer else -playerValue[0]/(state.game.numPlayers-1) for player in range(state.game.numPlayers)]
        
        return result
