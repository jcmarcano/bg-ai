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
                 numCPUs=None,
                 multiAgent=False):
        
        self.game = game
        self.multiAgent=multiAgent
        if self.multiAgent:
            self.samples = [GameSamplesManager(game, "alphazero" + "_" + self.game.playerRepr(player)) for player in range(self.game.numPlayers)]
        else:
            self.samples = [GameSamplesManager(game, "alphazero")]

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

        agents = []
        for player in range(len(self.samples)):
            provider = self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))
            provider.initModel()
            if self.multiAgent:
                provider.loadCheckpoint(repr(self.game), "alphazero_best_" + self.game.playerRepr(player))
                agentPlayer = player
            else:
                provider.loadCheckpoint(repr(self.game), "alphazero_best")
                agentPlayer = None

            agents.append(MCTSAgent(self.game, iterMax=self.iterMax, exploration=self.exploration, evaluator=AlphaZeroEvaluator(provider), player=agentPlayer))

        curPlayer = 0
        state:State = self.game.getInitState()
        
        episodeStep = 0

        observationTensors = []
        while not state.isTerminal():
            episodeStep += 1

            if state.isChanceState():
                setId = state.determinize(randomGenerator)
                for agent in agents:
                    agent.externalActionEvent(setId)
            else:
                observation = state.getObservation(curPlayer)
                logger.info(observation)
                print(observation)

                if self.multiAgent:
                    policy = agents[curPlayer].selectPolicy(observation)
                else:
                    policy = agents[0].selectPolicy(observation)

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
                observationTensors += [(obs[0], obs[1], curPlayer) for obs in observation.getObservationSymmetries(curPlayer, tuple(policyTensors))]

                if episodeStep < self.temperatureDrop:
                    selectedActionPos = randomGenerator.choice(len(policy),p=adjustedProbs)
                    action = policy[selectedActionPos][0]
                else:
                    action = max(policy, key=lambda action: action[1])[0]

                print(f"Episode: {episodeNumber}, Step {episodeStep}, Player {curPlayer}, Action {action}")
                logger.info(f"Episode: {episodeNumber}, Step {episodeStep}, Player {self.game.playerRepr(curPlayer)}, Action {state.actionRepr(action, curPlayer)}")

                state.performAction(action, curPlayer)
                for agent in agents:
                    agent.externalActionEvent(action, curPlayer)
                curPlayer = state.getNextPlayer()

        result = state.getResult() # Get Value for current Player

        # Return list of observations tensors for player 0 with policies and final value
        if self.multiAgent:
            return [[(observation[0], observation[1] + (result[observation[2]], )) for observation in observationTensors if observation[2] == player] for player in range(self.game.numPlayers)]
        else:
            return [[(observation[0], observation[1] + (result[observation[2]], )) for observation in observationTensors]]
    
    def selfPlay(self, iteration=None):
        if iteration:
            iter = iteration
        else:
            lastIter = self.samples[0].getLastIteration()

            if lastIter is None:
                iter = 0
            else:
                for s in self.samples:
                    s.loadTrainSamples(lastIter)    
                iter = lastIter + 1

        logger.info(f'Starting Iter #{iter + 1} ...')
        print(f'Starting Iter #{iter + 1} ...')

        if self.parallel:
            for samples in p_map(self.executeEpisode, range(self.selfPlayEpisodes), self.game.randomGenerator.spawn(self.selfPlayEpisodes), num_cpus = self.numCPUs):
                for player, s in enumerate(self.samples):
                    s.addSamples(samples[player])
        else:
            for episode in tqdm(range(self.selfPlayEpisodes)):
                samples = self.executeEpisode(episode, self.game.randomGenerator)
                for player, s in enumerate(self.samples):
                    s.addSamples(samples[player])

        for s in self.samples:
            s.saveTrainSamples(iter)

    def trainSamples(self, iteration=None):
        # load trainSamples
        if iteration:
            iter = iteration
        else:
            lastIter = self.samples[0].getLastIteration()
            if lastIter is None:
                logger.info("There are no samples to train")
                exit(1)
            
            iter = lastIter + 1

        logger.info(f'Train Phase for Iter #{iter} ...')

        provider = self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))
        for player, sm in enumerate(self.samples):
            bestModelName = "alphazero_best"
            tempModelName = "alphazero_temp"

            if self.multiAgent:
                bestModelName += "_" + self.game.playerRepr(player)
                tempModelName += "_" + self.game.playerRepr(player)

            provider.initModel()

            if not provider.loadCheckpoint(repr(self.game), bestModelName):
                provider.saveCheckpoint(repr(self.game), bestModelName)    

            
            sm.loadTrainSamples(lastIter)
            # TODO: shuffle samples before training

            # training new network, keeping a copy of the old one

            provider.train(sm.trainSamples)
            provider.saveCheckpoint(repr(self.game), tempModelName)


    def evaluateGame(self, evalPlayer, providers, randomGenerator, agentVersions=["temp","best"]):
        self.game.randomGenerator = randomGenerator
        agents = [None]*2
        for player, provider in enumerate(providers):
            provider.initModel()

            modelName = "alphazero_" + agentVersions[player] + ("_" + self.game.playerRepr(player) if self.multiAgent else "")
            if provider.loadCheckpoint(repr(self.game), modelName):
                agent = MCTSAgent(self.game, iterMax=self.iterMax, exploration=self.exploration, evaluator=AlphaZeroEvaluator(provider))
                if player == evalPlayer:
                    agents[0] = agent
                else:
                    agents[1] = agent
            else:
                raise Exception(f"No checkpoint found: {modelName} for game: {repr(self.game)}")

        # Create Arena
        arena = Arena(self.game, agents, display=True)

        # Play game
        result = arena.playGame()

        evalResult = [{},{}]
        for i in range(len(agentVersions)):
            if evalPlayer == 0:
                evalResult[i][agentVersions[i]] = result[i]
            else:
                evalResult[i][agentVersions[i]] = result[1-i]

        return evalResult

    def evaluate(self, iteration=None):
        # Perform episodes evaluation the model using new model as first player and then as second player
        providers = [self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,)),
                     self.provider(self.model, self.game.getObservationShapes(), self.game.getActionSize() + (1,))]

        if self.multiAgent:
            agentVersions = [["temp", "temp"], ["temp", "best"], ["best", "temp"], ["best", "best"]]
            totalResult = [{"temp": 0, "best": 0}, {"temp": 0, "best": 0}]
        else:
            agentVersions = [["temp", "best"]]
            totalResult = [{"temp": 0, "best": 0}]

        results = []

        if self.parallel:
            evalPlayerBase = [0] * len(agentVersions) + [1] * len(agentVersions)
            evalPlayerParam = evalPlayerBase * (self.evalEpisodes//(2*len(agentVersions))) + evalPlayerBase[:self.evalEpisodes%(2*len(agentVersions))]

            agentVersionParam = agentVersions * (self.evalEpisodes//4) + agentVersions[:self.evalEpisodes%4]
            for result in p_map(self.evaluateGame, 
                                            evalPlayerParam, 
                                            [providers]*self.evalEpisodes, 
                                            self.game.randomGenerator.spawn(self.evalEpisodes),
                                            agentVersionParam):
                results.append(result)
                for i in range(len(result)):
                    version = list(result[i].keys())[0]
                    if self.multiAgent:
                        totalResult[i][version] += result[i][version]
                    else:
                        totalResult[0][version] += result[i][version]
        else:
            evalPlayer = 0
            versionComb = 0
            for _ in tqdm(range(self.evalEpisodes)):
                result = self.evaluateGame(evalPlayer, 
                                           providers, 
                                           self.game.randomGenerator,
                                           agentVersions[versionComb])
                results.append(result)
                for i in range(len(result)):
                    version = list(result[i].keys())[0]
                    if self.multiAgent:
                        totalResult[i][version] += result[i][version]
                    else:
                        totalResult[0][version] += result[i][version]

                evalPlayer = 1 - evalPlayer
                versionComb = (versionComb + 1) % 4 if self.multiAgent else 0
        
        #compare models
        for player in range(len(totalResult)):
            bestModelName = "alphazero_best"
            tempModelName = "alphazero_temp"

            if self.multiAgent:
                bestModelName += "_" + self.game.playerRepr(player)
                tempModelName += "_" + self.game.playerRepr(player)

            if totalResult[player]["temp"] > totalResult[player]["best"]:
                # Temp model has better results, accept it as best model

                providers[0].renameCheckpoint(repr(self.game), tempModelName, bestModelName)
                print(f"Accepting new model for agent: {player}")
            else:
                providers[0].deleteCheckpoint(repr(self.game), tempModelName)
                print(f"Rejecting new model for agent: {player}")


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
