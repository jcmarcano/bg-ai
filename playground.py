import argparse
from bg_ai import Arena, Agent, LogManager, getDefaultRandomGenerator
import numpy as np
import importlib

logger = LogManager().getLogger("SandBox")

def parseParameters(paramList):
    params = {}
    for agentParam in paramList:
        paramParts = agentParam.split("=")
        if len(paramParts) != 2 or paramParts[1] == "":
            logger.error(f"Error parsing parameters. Expected a 'param=value' list. Received parameters: {paramList}")
            exit(1)
        if paramParts[1] == "None":
            params[paramParts[0]] = None
        elif paramParts[1] == "True":
            params[paramParts[0]] = True
        elif paramParts[1] == "False":
            params[paramParts[0]] = False
        else: 
            # Try to convert to Int
            try:
                params[paramParts[0]] = int(paramParts[1])
            except:
                # Try to convert to Float
                try:
                    params[paramParts[0]] = float(paramParts[1])
                except:
                    # leave param as is
                    params[paramParts[0]] = paramParts[1]

    return params

def play(game, agentSignatures=None, players=None, agentPackage="agents"):
    #Validate agents
    if not agentSignatures:
        agentSignatures = ["default"]

    if len(agentSignatures) == 1:
        if players is None and game.getPlayerRange()[0] != game.getPlayerRange()[1]:
            logger.error(f"Number of players is required for this game when using one agent, use -n option")
            exit(1)
        numPlayers = players if players else game.getPlayerRange()[0]
        agentSignatures *= numPlayers
    else:
        if len(agentSignatures) < game.getPlayerRange()[0] or len(agentSignatures) > game.getPlayerRange()[1]:
            logger.error(f"There are less or more agents than required players")
            exit(1)
        if players and players != len(agentSignatures):
            logger.error(f"A agent is required for each player in the game")
            exit(1)
        numPlayers = len(agentSignatures)

    game.numPlayers = numPlayers

    # Load Agent Modules and instantiate classes
    agentModules = dict()
    agents = []
    for playerId, signature in enumerate(agentSignatures):
        signatureParts = signature.split(":")
        agentName = signatureParts[0]
        
        #Get Agent parameters
        agentParams = parseParameters(signatureParts[1:])

        if agentName == "default" or agentName == "random":
            AgentClass = Agent
        else:
            if agentName not in agentModules:
                try:
                    agentModules[agentName] = importlib.import_module(agentPackage + "." + agentName)
                except ModuleNotFoundError:
                    logger.error(f"Agent module '{agentPackage}.{agentName}' not found")
                    exit(1)

            # Get Agent class and instantiate it
            agentClassNames = [gc for gc in dir(agentModules[agentName]) if gc.endswith("Agent") and gc != "Agent"]
            if len(agentClassNames) > 0:
                agentClassName = agentClassNames[0]
            else:
                agentClassNames = "Agent"

            try: 
                AgentClass = getattr(agentModules[agentName],agentClassName)
            except AttributeError:
                logger.error(f"Agent class not found in agent module '{agentPackage}.{agentName}'")
                exit(1)

        agents.append(AgentClass(game, playerId, **agentParams))

    # Create Arena
    arena = Arena(game, agents, display=True)

    # Play game
    logger.info(f"Init a game of {repr(game)} with {len(agents)} players")
    arena.playGame()

def train(game, trainerSignature, step=None, package="trainers"):
    signatureParts = trainerSignature.split(":")
    trainerName = signatureParts[0]
    trainerParams = parseParameters(signatureParts[1:])

    # Load Algorithm Module
    try:
        classModule = importlib.import_module(package + "." + trainerName)
    except ModuleNotFoundError:
        logger.error(f"Algorithm module '{package}.{trainerName}' not found")
        exit(1)

    # Get Algorithm class and instantiate it
        
    print(dir(classModule))
    trainerClassName = ""
    trainerClassNames = [tc for tc in dir(classModule) if tc.endswith("Trainer") and tc != "Trainer"]
    if len(trainerClassNames) > 0:
        trainerClassName = trainerClassNames[0]
    else: 
        logger.error(f"Algorithm class '{trainerClassName}' not found in game module '{package}.{trainerName}'")
        exit(1)

    TrainerClass = getattr(classModule, trainerClassName)
    trainer = TrainerClass(game, **trainerParams)
    
    ## Train trainer
    trainer.train(step)

if __name__ == "__main__":
    """ Play games using ai agents
        
    """
    gameParser = argparse.ArgumentParser(add_help=False)
    gameParser.add_argument("game", help="Name of the game module")
    gameParser.add_argument("parameters", help="game parameters in format 'param=value'", nargs="*")
    gameParser.add_argument("--g_package", help="Package of the game module", default="games", metavar="PACKAGE_NAME")
    gameParser.add_argument("-n", "--players", help="Num of players", default=None, type=int)
    gameParser.add_argument("-s", "--seed", help="Seed to initialize the random number generator", type=int)
    gameParser.add_argument("-v", "--verbose", help="Verbose output (-v: Info, -vv: Debug)", action="count", default=0)

    parser = argparse.ArgumentParser(description="Play games and train algorithms with AI")
    subParsers = parser.add_subparsers(help="select a command", title="commands", dest="cmd", required=True)

    playParser = subParsers.add_parser("play", help= "Play games using ai agents", parents=[gameParser])

    playParser.add_argument("-a", "--agent", help="Agent who will play the game. Use separate -a parameters to add individual agents. Agent Parameters are passed in format 'agent:param1=value1:param2=value2", 
                         action="append", dest="agentSignatures", metavar="AGENT_MODULE", nargs="?")
    playParser.add_argument("--a_package", help="Package of the Agent module", default="agents", metavar="PACKAGE")

    trainParser = subParsers.add_parser("train", help= "train a game using an AI algorithm", parents=[gameParser])
    trainParser.add_argument("-t", "--trainer", help="NAme of the trainer module. Training Parameters are passed in format 'trainer:param1=value1:param2=value2", 
                            dest="trainerSignature", metavar="TRAINER_MODULE")
    trainParser.add_argument("--t_package", help="Package of the Trainer module", default="trainers", metavar="PACKAGE")
    trainParser.add_argument("--step", help="Step of the trainer")

    args = parser.parse_args()

    # Set logging level
    logManager = LogManager()
    logManager.setVerbose(args.verbose)
    logger = logManager.getLogger("SandBox")

    # Load Game Module
    try:
        classModule = importlib.import_module(args.g_package + "." + args.game)
    except ModuleNotFoundError:
        logger.error(f"Game module '{args.g_package}.{args.game}' not found")
        exit(1)

    # Get Game class and instantiate it
    gameClassNames = [gc for gc in dir(classModule) if gc.endswith("Game") and gc != "Game"]
    if len(gameClassNames) > 0:
        gameClassName = gameClassNames[0]
    else:
        gameClassNames = "Game"

    try: 
        Game = getattr(classModule, gameClassName)
    except AttributeError:
        logger.error(f"Game class not found in game module '{args.g_package}.{args.game}'")
        exit(1)

    # Create unique Random Generator    
    randomGenerator = getDefaultRandomGenerator(args.seed)

    # Get Game parameters
    gameParams = parseParameters(args.parameters)

    game = Game(randomGenerator=randomGenerator, **gameParams)

    # Validate players
    if args.players and (args.players < game.getPlayerRange()[0] or args.players > game.getPlayerRange()[1]):
        logger.error(f"Invalid number of players")
        exit(1)

    if args.cmd=="play":
        play(game, args.agentSignatures, args.players)
    elif args.cmd=="train":
        train(game, args.trainerSignature, args.step, args.t_package)

