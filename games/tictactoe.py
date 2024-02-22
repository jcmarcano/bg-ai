from bg_ai import Game, State
import numpy as np

PLAYERS = "XO"
class TicTacToeGame(Game):
    pass
    
class TicTacToeState(State):
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player X, 2 = player 2 (O)
    """
    def initState(self):
        self.addComponent("board", initValue=np.zeros(9))
        
    def performAction(self, action, player):
        """ Update a state by carrying out the given action.
            Must update player.
        """
        if action not in range(9) or self.board[int(action)] != 0:
            print(self)
            print (action)
        assert action in range(9) and self.board[int(action)] == 0
        self.board[action] = player + 1
        self.lastPlayer = player
        
    def getValidActions(self, player):
        """ Get all possible action from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def isTerminal(self):
        return 1 in self.getResult() or np.count_nonzero(self.board) == 9
    
    def getResult(self):
        """ Get the game result
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] != 0 and self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == 1:
                    return [1, -1]
                else:
                    return [-1, 1]
        return [0, 0]

    def playerRepr(self, player):
        return PLAYERS[player]

    def actionRepr(self, action, player):
        return f"({action//3}, {action%3 + 1})"

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ("." + PLAYERS)[int(self.board[i])]
            if i % 3 == 2: s += "\n"
        return s
