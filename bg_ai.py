import numpy as np
from collections import deque
import copy
import logging
import sys
import re
import compress_pickle 
from pathlib import Path



class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. 
    """
    def __init__(self, numPlayers = None, randomGenerator=None, **kwargs):
        """
            Initialize the game Class. Do not overwrite it
        """
        self.numPlayers = self.getPlayerRange()[1] if numPlayers is None else numPlayers
        self.randomGenerator = randomGenerator if randomGenerator else getDefaultRandomGenerator()

        self._info = {}
        self._stateClass = self.getStateClass()
        self.initGame(**kwargs)


    def getStateClass(self):
        #Find state class
        mod = sys.modules[self.__module__]

        # Get State class name
        stateClassNames = [sc for sc in dir(mod) if sc.endswith("State") and sc != "State"]
        if len(stateClassNames) > 0:
            stateClassNames = stateClassNames[0]

            try: 
                return getattr(mod, stateClassNames)
            except AttributeError:
                pass

        raise("State Class Not Found. State Class must end with 'State'. Otherwise, Game Class must override getStateClass()")


    def getInitState(self, initPlayer=0):
        """
        get the init state from the State Class. Do not overwrite it
        Input:
            initPlayer. Initial player
        Returns:
            startState: the initial state of the game
        """
        state:State = self._stateClass(self, initPlayer) # Instantiate State Class
        state.initState()
        return state

    def initGame(self, **kwargs):
        """
        Abstract method used to initialize the game components
        """
        pass
        
    def getObservationShapes(self):
        """
        Returns:
            (o1, o2, .. on): a tuple of observation shapes.
                             e.g. ((7,7), 10) for a board of 7x7 and extrainfo of 10 positions
        """
        raise NotImplementedError()

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        raise NotImplementedError()
    
    def getPlayerRange(self):
        """
        Returns:
            players: a tuple (min, max) of accepted players
        """
        return (2, 2)

    def playerRepr(self, player):
        return str(player)
    

    def getAgentDefaults(self, agentName=""):
        """
        Input: 
            policyName: NAme of the policy
        Returns:
            kwargs: a dictionary of default keyword parameters to be used by the policy for this game
                    always end the method calling super().getPolicyDefaults()
        """
        return {}
    
    def getTrainerDefaults(self, trainerName=""):
        """
        Input: 
            policyName: NAme of the policy
        Returns:
            kwargs: a dictionary of default keyword parameters to be used by the policy for this game
                    always end the method calling super().getPolicyDefaults()
        """
        return 
    
        raise Exception (f"Trainer '{trainerName}' not available for game '{repr(self)}'")

    def __repr__(self):
        gameClass = self.__class__.__name__
        posGame = gameClass.rfind("Game")
        if posGame < 0:
            return gameClass
        else:
            return gameClass[:posGame]

class _PV:
    def __repr__(self):
        return "PRIVATE_VALUE"

class ComponentContainer():
    """
    This class is a container for Game Components. It generate access to each component as a property of the state, the observation or the player
    A class must inherit form this class
    """

    # Constants
    PRIVATE_ACCESS = 0
    PUBLIC_ACCESS = 1

    class _SimpleRepr(type):
        def __repr__(cls):
            return cls.__name__
        
    class PRIVATE_VALUE(metaclass = _SimpleRepr): pass

    def __init__(self, components=None):
        if components == None:
            self._components = {}
        else:
            self._components = components

    def __getattr__(self, name):
        """
        Get the value of a component, as a property of the inherited class
        """
        if (name in self._components):
            if self._components[name]["value"] is self.PRIVATE_VALUE:
                message = f"'cannot access private component '{name}'"
                raise AttributeError(message)
            return self._components[name]["value"]
        elif name in self.__dict__: 
            return self.__dict__[name]
        else:
            message = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(message)

    def __setattr__(self, name, value):
        """
        Set the value of a component, as a property of the inherited class
        """
        if name in ("_components", "PrivateValue"):
            object.__setattr__(self, "_components", value)
        elif (name in self._components):
            self._components[name]["value"] = value
        else:
            self.__dict__[name] = value

    def addComponent(self, componentName, access=PUBLIC_ACCESS, initValue=None):
        """
        Register a component in the inherited class
        """
        if access != self.PRIVATE_ACCESS and access != self.PUBLIC_ACCESS:
            raise Exception ("Invalid component access")

        if componentName in self._components:
            raise Exception ("component already registered")    
        self._components[componentName] = {
            "value": initValue,
            "access": access
        }

    def hidePrivateComponents(self, access = PUBLIC_ACCESS):
        newComponents = {}
        for component in self._components:
            if self._components[component]["access"]:
                newComponents[component] = copy.deepcopy(self._components[component])
            else:
                newComponents[component] = {
                    "value": self.PRIVATE_VALUE,
                    "access": self.PRIVATE_ACCESS
                }
        return ComponentContainer(newComponents)
            
    
    def hasPrivateComponents(self):
        for component in self._components:
            if self._components[component]["value"] is self.PRIVATE_VALUE:
                return True
            
        return False


    def __deepcopy__(self, memo):
        """
        Depp Copy the container
        """
        return self.__class__(copy.deepcopy(self._components))
    
    def asDict(self):
        return self._components
        

class State(ComponentContainer):
    """
    This class specifies an state of the the Game. To define your own game, subclass
    this class and implement the functions below. 

    """
    def __init__(self, game, initPlayer=0):
        """
            Initialize the state Class. Do not overwrite it
        """
        super().__init__()
        self.game = game
        self.lastPlayer = None
        self.initPlayer = initPlayer
        self.isObservation = False

        self._playerComponents = []
        for _ in range(self.game.numPlayers):
            self._playerComponents.append(ComponentContainer())

    def addComponent(self, componentName, access=ComponentContainer.PUBLIC_ACCESS, initValue=None, isPlayerComponent=False):
        """
        Register a component in the state, setting the access scope (Public or Private) for the state or for each player
        """
        if access != ComponentContainer.PRIVATE_ACCESS and access != ComponentContainer.PUBLIC_ACCESS:
            raise Exception ("Invalid component access")

        if isPlayerComponent:
            for container in self._playerComponents:
                container.addComponent(componentName, access, copy.deepcopy(initValue))
        else:
            super().addComponent(componentName, access, initValue)

    def __getitem__(self, index):
        """
        Access player components using state[player] its component container
        """
        return self._playerComponents[index]

    def initState():
        """
        Abstract method that returns the initial state of the game
        """
        raise NotImplementedError()
    
    def clone(self):
        """ Create a deep clone of this game state. All components of the state are cloned. In most of the cases there is not need to reimplement it, 
            if all state properties are stored as components
        """
        st = self.__class__(self.game)
        st.lastPlayer = self.lastPlayer
        st._components = copy.deepcopy(self._components)
        st._playerComponents = copy.deepcopy(self._playerComponents)

        return st

    def getValidActions(self, player):
        """
        Input:
            player: current player

        Returns:
            validActions: a list of valid actions for player for the current state
        """
        raise NotImplementedError()

    def performAction(self, action, player):
        """
        Abstract method that executes an action in the state
        Input:
            state: current state
            player: current player (1 or -1)
            action: action taken by current player

        """
        raise NotImplementedError()

    def getNextPlayer(self):
        """
        Get Next player of the game. Must be reimplemented if the game do not have a turn based game play
        Returns:
            nextPlayer: player who plays in the next turn
        """
        nextPlayer = self.lastPlayer + 1 if self.lastPlayer is not None else self.initPlayer
        return 0 if nextPlayer >= self.game.numPlayers else nextPlayer
    
    def getObservation(self, player):
        """
        Get a representation of the state, from the player point of observation. It contains only information of the player and public information of the state and other players
        """
        st = self.__class__(self.game)
        st.lastPlayer = self.lastPlayer
        st._components = self.hidePrivateComponents().asDict()
        
        playerComponents = []
        for p, container in enumerate(self._playerComponents):
            playerComponents.append(copy.deepcopy(container) if p == player else container.hidePrivateComponents())
        st._playerComponents = playerComponents

        st.isObservation = True
        st.observationPlayer = player
        return st

    def isTerminal(self):
        """
        Abstract method that returns if the state is a terminal state (someone wins or there is a tie)
        Returns:
            gameEnded: True if state is a terminal state, otherwise, false
        """
        raise NotImplementedError()

    def getResult(self):
        """
        Get the result of the state (usually a terminal state), as a list containing the result for each player in the game
        Returns:
            results: a list with the result of the game in te state for each player. e.g. [1, -1] means
                     that player 1 (won) has a result of 1 and player 0 a result of -1 (lose). 
                     0 is used for no results
               
        """
        raise NotImplementedError()

    def isChanceState(self):
        """
        IF the state generates random information return True, else False
        Returns:
            isChanceState: True is the state generates random information
        """
        return False

    def hasPrivateComponents(self):
        """
        Returns:
            hasPrivateComponents: True if the policy needs to infer values for private Information in this state
                              
        """
        if super().hasPrivateComponents():
            return True
        
        for container in self._playerComponents:
            if container.hasPrivateComponents():
                return True
            
        return False
    
    def generateRandomInformationSet(self, player=None):
        """
        Input:
            player: (optional) - Player who is observing the state

        Returns:
            Information Set State with random information for not known components of the Observation or for chance states.
            The generated random information must have valid values for the component and must consider all possible values for the state

        Remarks: 
            - All Private components MUST be set in this method. 
            - In case of chance States, this method MUST reset any information that fires 'isChanceState'
            - This method must not be called directly to prevent infinite loops
        """
        raise NotImplementedError()


    def determinize(self, player=None):
        """
        This method must be called from Polices to generate information sets that generates unique states. It makes the validations need to prevent infinite loops
        """
        setId = self.generateRandomInformationSet(player)
        if (setId is None):
            raise Exception ("Invalid SetId: {setId}")
        try:
            _ = setId < setId
        except:
            raise Exception ("'SetId' type must be comparable an sortable")
        if self.isChanceState():
            raise Exception("The state is still having the status of a Chance State")
        if self.hasPrivateComponents():
            raise Exception("The State has at least one private Component not set with valid values, after the creation of the Information Set")
        
        # The Observation is now a State
        self.isObservation = False
        self.observationPlayer = None

        return setId

    def getObservationTensor(self, player):
        """
        Input:
            player: player from whose point of view observations are generated.
            action_prob: policy vector with probabilities for each action in self.getActionSize()

        Returns:
            observations: a list of [(tensor,action_prob)] where each tuple is a symmetrical
                        form of the observation and the corresponding probabilities vector. This
                        is used when training algorithms with samples
        """
        raise NotImplementedError()

    def getObservationSymmetries(self, player, policyTuple):
        """
        Input:
            player: player from whose point of view observations are generated.
            policy: policy vector with probabilities for each action

        Returns:
            observations: a list of [(tensor,policy)] where each tuple is a symmetrical
                        form of the observation and the corresponding policy vector. This
                        is used when training algorithms with samples
        """
        raise NotImplementedError()


    def actionRepr(self, action, player):
        return str(action)


class Agent:
    """
    Abstract Class for an Agent. Implements selectAction that receives an observation (state) and returns an action
    A numpy random Generator is take from the game to share the same seed. It can be used with agent's policy estimator
    """
    def __init__(self, game, player=0, **kwargs):
        self.game:Game = game
        self.player = player

        # Get agent default parameters from the game

        self.kwargs = game.getAgentDefaults(self.__class__.__name__)
        if self.kwargs is None:
            self.kwargs = {}
        if kwargs:
            self.kwargs.update(kwargs)

        self.initAgent(**self.kwargs)

    def initAgent(*kwargs):
        pass

    def selectAction(self, observation):
        """
        Select an action. If not override, action is selected randomly
        """
        actions = observation.getValidActions(self.player)
        if len(actions) == 0:
            raise Exception(f"No actions available for player {self.game.playerRepr(self.player)}")
        
        randomAction = self.game.randomGenerator.choice(len(actions))
        return actions[randomAction]

    def externalActionEvent(self, action, player=None):
        """
        Process actions processed by any player (including the agents' player) or external events
        """
        pass

class Trainer:
    """
    Abstract Class for a Model Trainer. Implements 'train' method that run a complete or a step of the training cycle
    """
    def __init__(self, game, **kwargs):
        self.game:Game = game

        # Get trainer default parameters from the game

        self.kwargs = game.getTrainerDefaults(self.__class__.__name__)
        if kwargs:
            self.kwargs.update(kwargs)

        if self.kwargs is None:
            self.kwargs = {}

        self.initTrainer(**self.kwargs)

    def initTrainer(*kwargs):
        pass

    def train(self, step=None):
        raise NotImplementedError()



class Provider:
    def __init__(self, model, inputShapes, outputShapes, folder="/temp", **kwargs):
        self.model = model
        self.folder = folder
        self.inputShapes = inputShapes
        self.outputShapes = outputShapes
        self.checkpointExtension = ".tar"

    def initModel(self):
        self.model.compile()

    def train(self, samples):
        raise NotImplementedError()

    def predict(self, input):
        raise NotImplementedError()

    def saveCheckpoint(self, gameName, checkPointName, iteration=None):
        raise NotImplementedError()

    def loadCheckpoint(self, gameName, checkPointName, iteration=None):
        raise NotImplementedError()
    
    def getFileName(self, gameName, checkPointName, iteration=None, removeExtension=False):
        fileName = f"{gameName}_{checkPointName}"
        if iteration is not None:
            fileName += f"_{iteration}"
        if not removeExtension:
            fileName += self.checkpointExtension
        return fileName

    def deleteCheckpoint(self, gameName, checkPointName, iteration=None):

        file = Path(self.folder) / self.getFileName(gameName, checkPointName, iteration)
        file.unlink(missing_ok=True)

    def renameCheckpoint(self, gameName, checkPointFrom, checkPointTo, iteration=None):
        fileFrom = Path(self.folder) / self.getFileName(gameName, checkPointFrom, iteration)
        fileTo = Path(self.folder) / self.getFileName(gameName, checkPointTo, iteration)
        fileFrom.replace(fileTo)


        
class LogManager:
    def __new__(cls):
        """ creates a singleton object, if it is not created, 
        or else returns the previous singleton object"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(LogManager, cls).__new__(cls)
        return cls.instance
    
    def __init__(self):
        if "_LogManager__initialized" in dir(self): return

        self.levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        self.level = logging.WARNING
        self.__initialized = True

    def getLogger(self, namespace):
        logger = logging.getLogger(namespace)
        logger.setLevel(self.level)
        return logger

    def setVerbose(self, verbose):
        self.level = self.levels[min(verbose, len(self.levels) - 1)]
        logging.basicConfig(level=self.level)


class Arena():
    logger = LogManager().getLogger(__name__)
    
    def __init__(self, game, agents, display=True):
        self.game = game
        self.agents = agents
        self.display = display

    def playGame(self, initPlayer = 0):        
        curPlayer = initPlayer
        state:State = self.game.getInitState()
        it = 0
        while not state.isTerminal():
            it += 1

            if state.isChanceState():
                setId = state.determinize(self.game.randomGenerator)
                for i in range(len(self.agents)):
                        self.agents[i].externalActionEvent(setId)
            else:
                observation = state.getObservation(curPlayer)
                if self.display:
                    print(observation)

                action = self.agents[curPlayer].selectAction(observation)
                self.logger.info(f"Turn {it}, Player {self.game.playerRepr(curPlayer)}, Action {state.actionRepr(action, curPlayer)}")

                state.performAction(action, curPlayer)

                for i in range(len(self.agents)):
                        self.agents[i].externalActionEvent(action, curPlayer)


                curPlayer = state.getNextPlayer()

        result = state.getResult()
        if self.display:
            print(state)
            winnerValue = max(result)
            winners = [self.game.playerRepr(player) for player, value in enumerate(result) if value == winnerValue]
            print(f"Game over: Turn {it}, {'Tie' if len(winners) == len(result) else 'Winner: ' + winners[0] if len(winners) == 1 else 'Winners: ' + str(winners)}")

        return result



class GameSamplesManager:
    logger = LogManager().getLogger(__name__)

    def __init__(self, game, modelName, folder="/temp", queueSize=100000):
        print(modelName)
        self.game = game
        self.modelName = modelName
        self.folder = folder
        self.trainSamples = deque([], queueSize)

    def getFileName(self, iteration=None, removeExtension=False):
        fileName = f"{repr(self.game)}_{self.modelName}_samples"
        if iteration is not None:
            fileName += f"_{iteration}"
        if not removeExtension:
            fileName += ".gz"
        return fileName

    def saveTrainSamples(self, iteration=None):
        file = Path(self.folder) / self.getFileName(iteration)
        print(file)
        compress_pickle.dump(self.trainSamples, file)
        print(f"saved on {file}")

    def loadTrainSamples(self, iteration=None):
        file = Path(self.folder) / self.getFileName(iteration)
        self.trainSamples = compress_pickle.load(file)

        return iteration

    def getLastIteration(self):
        iteration = None

        for file in Path(self.folder).iterdir():
            if file.name.startswith(self.getFileName(removeExtension=True)):
                iterRegExp = self.getFileName("([0-9]+)")
                fa = re.findall(iterRegExp, file.name)
                if len(fa) > 0:
                    if iteration is None or int(fa[0]) > iteration:
                        iteration = int(fa[0])
        return iteration


    def addSamples(self, samples):
        for sample in samples:
            self.trainSamples.append(sample)
        

## Utils
def getDefaultRandomGenerator(seed=None):
    return np.random.default_rng(seed)

