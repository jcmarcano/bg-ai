from bg_ai import Game, State 

class FourInRowGame(Game):
    def initGame(self, length = 7, height = 6, version = 4):
        assert length == int(length) and length > 4 and height == int(height) and height % 2 == 0 and version in [4,5] # length and height must be integer, lg grater than 4 and hg pair.
        self.length = length
        self.height = height
        self.version = version

    def getAgentDefaults(self, agentName):
        if agentName == "MCTSAgent":
            return {
                "iterMax": 5000
            }
        if agentName == "HumanAgent":
            return {
                "actionsAsOptions":True,
                "hideOptions": True
            }
        return super().getAgentDefaults()
    
    def playerRepr(self, player):
        return "XO"[player]


class FourInRowState(State):
    def initState(self): 
        self.addComponent("board", initValue=[])

        if (self.game.version == 5):
            self.board.append([1,2]*int(self.game.height/2))  ## Used in 5 in a Row variant
            self.game.length += 1
        for _ in range(1,self.game.length + 1):
            self.board.append([0]* self.game.height)
        if (self.game.version == 5):
            self.board.append([2,1]*int(self.game.height/2)) ## Used in 5 in a Row variant
            self.game.length += 1
            
    def performAction(self, action, player):
        """ Update a state by carrying out the given action.
            Must update player.
        """
        pos = self.board[action].index(0)
        assert action >= 0 and action < self.game.length and action == int(action) and pos >= 0
        self.board[action][pos] = player + 1
        self.lastPlayer = player
        
    def getValidActions(self, player):
        """ Get all possible actions from this state.
        """
        return [i for i in range(self.game.length) if 0 in self.board[i]]
        
    def isTerminal(self):
        return 1 in self.getResult() or [i for i in range(self.game.length) if 0 in self.board[i]] == []
    
    def getResult(self):
        verbose = False
        if verbose: print("VERTICAL")
        for x in range(self.game.length):
            lastValue = -1
            equalCount = 0
            for y in range(self.game.height):
                value = self.board[x][y]
                if value > 0 and (value == lastValue or lastValue == -1):
                    equalCount += 1
                else:
                    equalCount = 0 if value == 0 else 1
                lastValue = value
                if equalCount >= self.game.version and lastValue:
                    if lastValue == 1: 
                        return [1, -1]
                    else: 
                        return [-1, 1]
                if verbose: print(x,y, equalCount, lastValue)

        if verbose: print("-----------------------------------------\nHORIZONTAL")

        for y in range(self.game.height):
            lastValue = -1
            equalCount = 0
            for x in range(self.game.length):
                value = self.board[x][y]
                if value > 0 and (value == lastValue or lastValue == -1):
                    equalCount += 1
                else:
                    equalCount = 0 if value == 0 else 1
                lastValue = value
                if equalCount >= self.game.version:
                    if lastValue == 1: 
                        return [1, -1]
                    else: 
                        return [-1, 1]
                if verbose: print(x,y, equalCount, lastValue)

        if verbose: print("-----------------------------------------\nDIAGONAL 1")

        for x in range(self.game.length - self.game.version + 1):
            lastValue1 = -1
            equalCount1 = 0
            lastValue2 = -1
            equalCount2 = 0
            for y in range(min(self.game.height, self.game.length - x)):
                value1 = self.board[x+y][y]
                if value1 > 0 and (value1 == lastValue1 or lastValue1 == -1):
                    equalCount += 1
                else:
                    equalCount = 0 if value1 == 0 else 1
                lastValue1 = value1
                value2 = self.board[x+y][self.game.height - 1 - y]
                if value2 > 0 and (value2 == lastValue2 or lastValue2 == -1):
                    equalCount += 1
                else:
                    equalCount = 0 if value2 == 0 else 1
                lastValue2 = value1
                if equalCount1 >= self.game.version and not verbose:
                    if lastValue == 1: 
                        return [1, -1]
                    else: 
                        return [-1, 1]
                if equalCount2 >= self.game.version and not verbose:
                    if lastValue == 1: 
                        return [1, -1]
                    else: 
                        return [-1, 1]
                if verbose: print("1: ", x+y,y, equalCount1, lastValue1, "           2: ", x+y,self.game.height - 1 - y, equalCount2, lastValue2)

        if verbose: print("-----------------------------------------\nDIAGONAL 2")

        for y in range(1, self.game.height - self.game.version + 1):
            lastValue1 = -1
            equalCount1 = 0
            lastValue2 = -1
            equalCount2 = 0
            for x in range(min(self.game.length, self.game.height - y)):
                value1 = self.board[x][x+y]
                if value1 > 0 and (value1 == lastValue1 or lastValue1 == -1):
                    equalCount += 1
                else:
                    equalCount = 0 if value1 == 0 else 1
                lastValue1 = value1
                value2 = self.board[x][self.game.height - 1 - y - x]
                if value2 > 0 and (value2 == lastValue2 or lastValue2 == -1):
                    equalCount += 1
                else:
                    equalCount = 0 if value2 == 0 else 1
                lastValue2 = value1
                if equalCount1 >= self.game.version and not verbose:
                    if lastValue == 1: 
                        return [1, -1]
                    else: 
                        return [-1, 1]
                if equalCount2 >= self.game.version and not verbose:
                    if lastValue == 1: 
                        return [1, -1]
                    else: 
                        return [-1, 1]
                if verbose: print("1: ", x,x+y, equalCount1, lastValue1, "           2: ", x,self.game.height - 1 - y - x, equalCount2, lastValue2)
        return [0,0]

    def actionRepr(self, action, player):
        return f"{action + 1}"

    def __repr__(self):
        s= ""
        for y in range(self.game.height - 1,-1,-1):
            for x in range(self.game.length):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s
