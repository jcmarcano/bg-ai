import numpy as np
import math
from bg_ai import State, LogManager

logger = LogManager().getLogger(__name__)

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of player.
    """

    def __init__(self, action, player=None, prevProb=None, isInfoSetNode = False, parent=None):
        self.action = action  # the action that got us to this node - "None" for the root node
        self.prevProb = prevProb
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.value = None
        self.visits = 0
        self.player = player
        self.isInfoSetNode = isInfoSetNode
        self.decisionNodes = []

    def getUntriedActions(self, legalActions = None):
        """ Return the elements of legalActions for which this node does not have children.
        """

        # Return all actions that are legal but have not been tried yet
        return [child.action for child in self.childNodes if child.visits == 0]

    def addChild(self, action, player=None, prevProb=None, isInfoSetNode=False):
        """ Add a new child node for the action m.
            Return the added child node
        """
        n = Node(action, player, prevProb, isInfoSetNode, parent=self)
        self.childNodes.append(n)
        return n
    
    def isChanceNode(self):
        return len(self.decisionNodes) > 0

    def update(self, result, lastDecisionNode):
        """ Update this node - increment the visit count by one, and increase the value by the result of terminalState
        """
        self.visits += 1
        if self.value is None:
            self.value = np.array(result)
        else:
            self.value += np.array(result)
        parentNode:Node = self.parentNode

        # Summarize actions under chance nodes
        if lastDecisionNode and self.isInfoSetNode and parentNode:
            action = lastDecisionNode.action

            decisionNode:Node = next((child for child in parentNode.decisionNodes if child.action == action), None)   
            if decisionNode is None:
                decisionNode = Node(action, lastDecisionNode.player, parent = parentNode)
                decisionNode.visits = 1
                decisionNode.value = np.array(result)
                parentNode.decisionNodes.append(decisionNode)
            else: 
                decisionNode.visits += 1
                decisionNode.value += np.array(result)

        # Update only actions, not setId's
        if self.isInfoSetNode:
            return lastDecisionNode
        else:
            return self


    def __repr__(self):
        return f"[ACT:{self.action} PLAY:{self.player} VAL/PROB/VIS: {self.value}/{self.prevProb}/{self.visits}] CN/ISN: {self.isChanceNode()}/{self.isInfoSetNode}"

    def treeToString(self, indent=0, depth=100):
        """ Represent the tree as a string, for debugging purposes.
        """
        s = self.indentString(indent) + str(self)
        if depth > 0:
            for c in self.childNodes:
                s += c.treeToString(indent + 1, depth - 1)
        return s

    def indentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def childrenToString(self):
        s = "\n"
        sortedChildNodes = self.childNodes.copy()
        sortedChildNodes.sort(key=lambda c: (c.visits, int(c.action)), reverse=True)
        if self.isChanceNode():
            sortedDecisionNodes = self.decisionNodes.copy()
            sortedDecisionNodes.sort(key=lambda c: c.visits, reverse=True)
            for c in sortedDecisionNodes:
                s += str(c) + "\n"
            s += "SETS: " + str(sortedChildNodes) + "\n"
        else:
            for c in sortedChildNodes:
                s += str(c) + "\n"
        return s
    
    def branchToString(self):
        nodes = []
        node = self
        while node.parentNode is not None:
            nodes.append(node)
            node = node.parentNode
        
        s=""
        for i in range(len(nodes)):
            s += self.indentString(i) + str(nodes.pop())

        return s

class simulatorEvaluator():
    def prevPolicy(self, state:State):
        """ Returns an equal distribution of probabilities for each valid action"""
        actions = state.getValidActions(state.getNextPlayer())
        # shuffle action to generate more variance
        return [(action,1/len(actions)) for action in actions]

    def nodeEvaluator(self, state: State):
        while not state.isTerminal():  # while state is non-terminal
            nextPlayer = state.getNextPlayer()
            if state.isChanceState() or state.hasPrivateComponents():
                # for a Chance Node, select a random action based on distribution probabilities
                state.determinize(nextPlayer)
            else:
                actions = state.getValidActions(nextPlayer)
                action = actions[state.game.randomGenerator.choice(len(actions))]
                state.performAction(action, nextPlayer)
        return state.getResult()


def uctSelector(node: Node, exploration=0.7):
    """ Use the UCT formula to select a child node
        exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
    """

    # Get the child with the highest UCT score
    s = max(
        node.childNodes,
        key=lambda c: float(c.value[c.player]) / float(c.visits) + exploration * math.sqrt(math.log(node.visits) / float(c.visits)),
    )

    # Return the child selected above
    return s

def puctSelector(node: Node, exploration=0.7):
    """ Use the PUCT formula to select a child node
        exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
    """
    #    print(node)
    # Get the child with the highest PUCT score
    s = max(
        node.childNodes,
        key=lambda c: float(c.value[c.player]) / float(c.visits) + exploration * c.prevProb * math.sqrt(node.visits) / (float(c.visits) + 1),
    )

    # Return the child selected above
    return s
    
class MCTS:
    def __init__(self, exploration=0.7, iterMax=1000, evaluator=simulatorEvaluator(), childSelector=puctSelector):
        self.exploration = exploration
        self.iterMax = iterMax
        self.evaluator = evaluator
        self.childSelector = childSelector

    def search(self, rootState: State, rootNode:Node = None, getRootNode=False):
        """ Conduct an MCTS search for iterMax iterations starting from rootState.
            Returns
              actions: list with actions and visits for each action
              rootNode: 
        """
        if rootNode is None:
            rootNode:Node = Node(None)

        for i in range(self.iterMax):
            state:State = rootState.clone()
            # initialize the private information, If the game handles it
#            state.resetPrivateInfo()

            node:Node = rootNode

            # Select
            while state.isChanceState() or state.hasPrivateComponents() or (not state.isTerminal() and node.getUntriedActions() == []):  
                if state.isChanceState() or state.hasPrivateComponents():
                    # for a Chance Node or states with private information, complete the state with random information
                    setId = state.determinize(state.getNextPlayer())

                    # return node if exists, otherwise create the node
                    setNode:Node = next((child for child in node.childNodes if child.action == setId), None)   
                    if setNode is None:
                        node = node.addChild(setId, isInfoSetNode=True)
                    else:
                        node = setNode

                elif node.childNodes == []:
                    # Create new node
                    actionsProbs = self.evaluator.prevPolicy(state)
                    # Shuffle action to have more variance
                    state.game.randomGenerator.shuffle(actionsProbs)

                    for action, prob in actionsProbs:
                        node.addChild(action, state.getNextPlayer(), prob)
                else:
                    # node is fully expanded, select child
                    node = self.childSelector(node, self.exploration)
                    assert node.action in state.getValidActions(node.player)  #Remove
                    state.performAction(node.action, node.player)
                            


            # Expand: Visit untried Action
            untriedActions = node.getUntriedActions()
            if untriedActions != []:  
                # if we can expand (i.e. state/node is non-terminal)

                action = untriedActions[state.game.randomGenerator.choice(len(untriedActions))] #  using state.game.randomGenerator.choice(untriedActions) will change the type of the action
                node = next(child for child in node.childNodes if child.action == action)
                if action not in state.getValidActions(state.getNextPlayer()):
                    print (state)
                    print (action)
                    print (node.branchToString())
                assert action in state.getValidActions(state.getNextPlayer())  # Remove
                state.performAction(action, state.getNextPlayer())

            # Evaluate node
            result = self.evaluator.nodeEvaluator(state)

            # Backpropagate
            lastDecisionNode = None if node.isInfoSetNode else node
            while node is not None:  
                # backpropagate from the expanded node and work back to the root node, using actions from decision nodes
                lastDecisionNode = node.update(result, lastDecisionNode)
                node:Node = node.parentNode

        # Output some information about the tree - can be omitted
#        logger.info(rootNode.childrenToString())
        logger.debug(rootNode.treeToString(depth = 3))
        logger.info(rootNode.childrenToString())

        if rootNode.isChanceNode():
            # for Chance Nodes, return the summarization of actions in decisions nodes under the chance node
            result = [(c.action, c.visits) for c in rootNode.decisionNodes]
        else:
            # return child actions 
            result = [(c.action, c.visits) for c in rootNode.childNodes]

        if getRootNode:
            return result, rootNode
        else:
            return result


