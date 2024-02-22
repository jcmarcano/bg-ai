from bg_ai import Agent, LogManager
from algorithms.mcts import MCTS
import numpy as np

logger = LogManager().getLogger(__name__)

class MCTSAgent(Agent):
    def initAgent(self, preserveRoot = True,  **kwargs):
        self.mcts = MCTS(**kwargs)
        self.rootNode = None
        self.preserveRoot = preserveRoot

    def selectAction(self, observation):
        policy = self.selectPolicy(observation)
        selectedAction = max(policy, key=lambda action: action[1])
        return selectedAction[0]
    
    def selectPolicy(self, observation):
        policy, self.rootNode = self.mcts.search(observation, rootNode=self.rootNode, getRootNode=self.preserveRoot)
        return policy

    def externalActionEvent(self, action, player=None):
        if self.rootNode is None:
            return
        if self.rootNode.childNodes == []:
            # Unable to preserve first node. The node was not exploded
            self.rootNode = None
            return

        for node in self.rootNode.childNodes:
            if node.isInfoSetNode:
                # Unable to preserve first node. Node is an non Chance Information Set
                self.rootNode = None
                return
            if node.player == player and node.action == action:
                self.rootNode = node
                return

        ## Unable to preserveNode, reset rootNode
        logger.info(f"Node action not found for player: {player} and action: {action}")

        logger.debug(f"children: {len(node.childNodes)}")
        logger.debug(self.rootNode.treeToString(depth=3))
        exit()
        self.rootNode = None

