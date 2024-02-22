from bg_ai import Game, State
from providers.keras import KerasProvider
import numpy as np


# Constants
ORIENTATIONS = 8

ORIENTATION_TYPES = [
    [0],                # 0 -> + like 
    [0,1],              # 1 -> | like
    [0,1,2,3],          # 2 -> U like
    [0,1,4,5],          # 3 -> ʃ like
    [0,1,2,3,4,5,6,7],  # 4 -> Irregular Patches
    [0,1,7]             # 5 -> Patch 0 + Advance Action
]

#Patch Config
P_FIGURE           = 0
P_TIME             = 1
P_PRICE            = 2
P_VALUE            = 3
P_ORIENTATION_TYPE = 4

#####################
#     PATCHWORK     #
#####################

# patches
# 0 figure
# 1 Time
# 2 Price
# 3 Value
# 4 Orientation Type (flip + rotation)

_PATCHES_FULL = [
    [[[1,1]],      1,2,0,1],  # 1
    [[[0,1],     
      [1,1]],      3,1,0,2],  # 2
    [[[0,1],     
      [1,1]],      1,3,0,2],  # 3
    [[[1,1],     
      [1,1]],      5,6,2,0],  # 4
    [[[1,1,1]],    2,2,0,1],  # 5
    [[[1,1,1],   
      [0,0,1]],    6,4,2,4],  # 6
    [[[0,0,1],   
      [1,1,1]],    2,4,1,4],  # 7
    [[[1,0,1],   
      [1,1,1]],    2,1,0,2],  # 8
    [[[1,1,1],
      [1,1,0]],    2,2,0,4],  # 9
    [[[1,1,0],
      [0,1,1]],    2,3,1,3],  #10
    [[[0,1,1],
      [1,1,0]],    6,7,3,3],  #11
    [[[0,1,0],   
      [1,1,1]],    2,2,0,2],  #12
    [[[1,0,0],
      [1,1,0],
      [0,1,1]],    4,10,3,2], #13
    [[[0,1,0],
      [0,1,0],
      [1,1,1]],    5,5,2,2],  #14
    [[[0,1,0],
      [1,1,1],
      [0,1,0]],    4,5,2,0],  #15
    [[[1,0,1],
      [1,1,1],
      [1,0,1]],    3,2,0,1],  #16
    [[[0,1,0],
      [1,1,1],
      [1,0,1]],    6,3,2,2],  #17
    [[[0,1,1],
      [0,1,1],
      [1,1,0]],    6,8,3,4],  #18
    [[[1,1,1,1]],  3,3,1,1],  #19
    [[[1,0,0,1],   
      [1,1,1,1]],  5,1,1,2],  #20
    [[[0,1,1,0],   
      [1,1,1,1]],  4,7,2,2],  #21
    [[[0,1,1,1],
      [1,1,1,0]],  2,4,0,3],  #22
    [[[0,0,1,0],
      [1,1,1,1]],  4,3,1,4],  #23
    [[[1,1,1,1],
      [0,0,1,1]],  2,3,1,4],  #24
    [[[1,0,0,0],   
      [1,1,1,1]],  3,10,2,4], #25
    [[[1,1,1,1],   
      [1,1,0,0]],  5,10,3,4], #26
    [[[0,1,1,0],
      [1,1,1,1],
      [0,1,1,0]],  3,5,1,1],  #27
    [[[1,0,0,0],
      [1,1,1,1],
      [1,0,0,0]],  2,7,2,2],  #28
    [[[0,1,0,0],
      [1,1,1,1],
      [0,1,0,0]],  3,0,1,2],  #29
    [[[0,0,0,1],
      [1,1,1,1],
      [1,0,0,0]],  2,1,0,3],  #30
    [[[0,1,0,0],
      [1,1,1,1],
      [0,0,1,0]],  1,2,0,3],  #31
    [[[1,1,1,1,1]],1,7,1,1],  #32
    [[[0,0,1,0,0],
      [1,1,1,1,1],
      [0,0,1,0,0]],4,1,1,1],  #33
]

_BOARD_LEATHER_PATCHES_FULL = [21, 27, 33, 45, 51]
_BOARD_BUTTONS_FULL = [6, 12, 18, 24, 30, 36, 42, 48, 54]


#####################
# PATCHWORK EXPRESS #
#####################

# patches
# - figure
# - Time
# - Price
# - Value
# - Orientation Type (flip + rotation)

_PATCHES_EXPRESS = [
    # Colorful patches
    [[[0,1],     
      [1,1]],      1,1,0,2],  # 1
    [[[1,1,1],   
      [0,0,1]],    3,1,0,4],  # 2
    [[[1,0,1],   
      [1,1,1]],    1,2,0,2],  # 3
    [[[1,1,1],
      [0,1,1]],    2,5,1,4],  # 4
    [[[1,1,0],
      [0,1,1]],    3,2,1,3],  # 5
    [[[0,0,1],
      [1,1,1]],    3,6,2,4],  # 6
    [[[1,0,0],
      [1,1,0],
      [0,1,1]],    3,0,0,2],  # 7
    [[[1,0,0],
      [1,1,1],
      [0,0,1]],    1,5,1,3],  # 8
    [[[0,1,0],
      [1,1,1],
      [1,0,0]],    4,2,1,4],  # 9
    [[[0,0,1],
      [0,0,1],
      [1,1,1]],    3,4,1,2],  #10
    [[[1,0,0],
      [1,1,1],
      [1,0,0]],    3,3,1,2],  #11
    [[[0,1,0],
      [1,1,1],
      [0,1,0]],    4,6,2,0],  #12
    [[[0,0,1,0],
      [1,1,1,1]],  1,3,0,4],  #13
    [[[0,1,1,1],
      [1,1,0,0]],  2,4,1,4],  #14
    [[[1,1,1,1,1]],1,2,0,1],  #15

    # Blue patches
    [[[1,1]],      2,0,1,1],  #16
    [[[1,1]],      1,0,0,1],  #17
    [[[1,1]],      2,4,2,1],  #18
    [[[1,1],
      [1,1]],      1,3,0,0],  #19
    [[[0,1],
      [1,1]],      2,2,1,2],  #20
    [[[1,1,1]],    3,0,1,1],  #21
    [[[1,1,1]],    1,2,0,1],  #22
    [[[1,1,1],
      [0,1,0]],    2,3,1,2]   #23
]

BLUE_PATCHES_INIT_POS = 16

_BOARD_LEATHER_PATCHES_EXPRESS = [8, 12, 16, 20, 24, 28]
_BOARD_BUTTONS_EXPRESS = [6, 10, 14, 18, 22, 26, 30]

PATCH_ACTIONS = 3

class PatchworkGame(Game):
    def initGame(self, version = 0):  # version: 0 = Full,1 = Express
        assert version in [0,1]
        self.version = version
        if self.version == 0:
            self.size = 9
            self.leatherPatches = _BOARD_LEATHER_PATCHES_FULL
            self.boardButtons = _BOARD_BUTTONS_FULL
            self.patches = _PATCHES_FULL
        else:
            self.size = 7
            self.leatherPatches = _BOARD_LEATHER_PATCHES_EXPRESS
            self.boardButtons = _BOARD_BUTTONS_EXPRESS
            self.patches = _PATCHES_EXPRESS


    def getAgentDefaults(self, agentName):
        if agentName == "MCTSAgent":
            return {
                "iterMax": 1000
            }
        if agentName == "AlphaZeroAgent":
            return {
                "iterMax": 300,
                "model": self.getModel(),
                "provider": KerasProvider
            }
        return super().getAgentDefaults()

    def getTrainerDefaults(self, agentName):
        if agentName == "AlphaZeroTrainer":
            return {
                "iterMax": 100,
                "selfPlayEpisodes": 10,
                "model": self.getModel(),
                "provider": KerasProvider
            }
        return super().getTrainerDefaults()


    def getObservationShapes(self):
        return ((self.size, self.size), 6 + len(self.patches))
    
    def getActionSize(self):
        return ((PATCH_ACTIONS + 1) * ORIENTATIONS * self.size*self.size, )
    
    def getModel(self):
        return PatchworkAlphaZeroModel(self.size, len(self.patches))
    
    def __repr__(self):
        if self.version == 0:
            return "Patchwork"
        else:
            return "PatchworkExpress"

class PatchworkAction:
    def __init__(self, size, patchPos, orientation, position):
        self.size = size
        self.patchPos = patchPos
        self.orientation = orientation
        self.position = position

    def __int__(self):
        return (self.patchPos    * ORIENTATIONS * self.size * self.size + 
                self.orientation * self.size * self.size +      
                self.position[0]   * self.size +    
                self.position[1])

    @staticmethod
    def fromStandardAction(size, action):
        patchPos = action // (ORIENTATIONS * size * size)
        orientation = (action % (ORIENTATIONS * size * size)) // (size * size)
        pos = action % (size * size)
        position = (pos // size, pos % size)

        return PatchworkAction(size, patchPos, orientation, position)

    def __repr__(self):
        return repr((self.patchPos, self.orientation, self.position))

    def __eq__(self, other):
        return self.patchPos == other.patchPos and self.orientation == other.orientation and self.position == other.position

class PatchworkState(State):
    def initState(self):
        # Player Boards
        self.addComponent("board",           initValue = np.zeros((self.game.size, self.game.size)), isPlayerComponent=True)
        self.addComponent("pos",             initValue=1,                                            isPlayerComponent=True)
        self.addComponent("buttons",         initValue=5,                                            isPlayerComponent=True)
        self.addComponent("value",           initValue=0,                                            isPlayerComponent=True)
        self.addComponent("nextButtonIndex", initValue = 0,                                          isPlayerComponent=True)

        # Main board
        self.addComponent("nextLeatherPatchIndex", initValue = 0)
        self.addComponent("bonus7X7", initValue = None)

        # Patches
        if self.game.version == 1:
            patchDeck = np.arange(2, 15 + 1)
        else:
            patchDeck = np.arange(2, len(self.game.patches) + 1)
        self.game.randomGenerator.shuffle(patchDeck)
        # Append Patch #1 at the end of the deck
        self.addComponent("patchDeck", initValue = np.append(patchDeck, [1])) 


    def performAction(self, action, player):
        patchPos = action.patchPos
        orientation = action.orientation
        position = action.position

#        print(f"player: {player}, patchNumber: {patchNumber}, orientation: {orientation}, position: {position}")

        board = self[player].board
        if patchPos == 0: 
            if orientation in [0,1]: # place a leather patch
                if orientation == 0: 
                    # place the patch
                    self[player].board[position[0], position[1]]  = 1
                    # otherwise, discard patch (orientation = 1)

                self.nextLeatherPatchIndex += 1
                self.lastPlayer = player
                return

            else:  # Advance to take buttons (orientation = 7)
#            print(self)
                spaces =  self[1 - player].pos - self[player].pos
                if (self[1 - player].pos < self.game.boardButtons[-1]): spaces +=1

                if spaces <= 0:
                    print("-------------------------------------------------------")
                    print(f"player: {player}, patchNumber: {patchNumber}, orientation: {orientation}, position: {position}")
                    print(self)
                    print(f"spaces: {spaces}")
                    print(f"result: {self.GetResult(player)}")
                    print("-------------------------------------------------------")
                    assert(spaces > 0)

                # Take buttons
                self[player].buttons += spaces

                # Advance player pawn
                self[player].pos += spaces

        else:  # Place Patch
            patchNumber = self.patchDeck[patchPos - 1]
            patch = self.game.patches[patchNumber - 1]
            patchFigure = self.getPatchOrientation(np.array(patch[P_FIGURE]), orientation)

            (patchHeight, patchWidth) = patchFigure.shape
#            print (board)
#            print (patchFigure)
#            print (patchHeight, patchWidth)
            board[position[0] : position[0] + patchHeight, position[1] : position[1] + patchWidth] += patchFigure
            if np.any(board [position[0] : position[0] + patchHeight, position[1] : position[1] + patchWidth] > 1):
                print(action)
                print(board)
                print(patchFigure)
            assert np.all(board [position[0] : position[0] + patchHeight, position[1] : position[1] + patchWidth] < 2)

            # Pay for patch
            self[player].buttons -= patch[P_PRICE]
            
            # Update player value
            self[player].value += patch[P_VALUE]

            # Advance player pawn
            self[player].pos += patch[P_TIME]

            # Remove patch from deck
#            print (self.patchDeck)
#            print (f"Remove patch {patchNumber} in pos {patchPos}")

            self.patchDeck = np.append(self.patchDeck[patchPos + 1:],self.patchDeck[:patchPos])

        # Validate 7x7 Bonus
        if self.game.version == 0 and self.bonus7X7 is None:
            for i in range(3):
                for j in range(3):
                    if np.count_nonzero(self[player].board[i:i+7, j:j+7]) == 49:
                        self.bonus7X7 == player

        # Update buttons when passing button mark
        if self[player].pos >= self.game.boardButtons[self[player].nextButtonIndex]:
            self[player].buttons += self[player].value
            self[player].nextButtonIndex += 1

        # if action is the last one, update button points
        if self[player].pos >= self.game.boardButtons[-1]:
            self[player].pos = self.game.boardButtons[-1]
            self[player].buttons -= (len(np.argwhere(board == 0)) * 2)
            self[player].buttons += 7 if self.bonus7X7 == player else 0

        self.lastPlayer = player
#        print (self)

    def getValidActions(self, player):

        if self.nextLeatherPatchIndex < len(self.game.leatherPatches) and self[player].pos >= self.game.leatherPatches[self.nextLeatherPatchIndex]:
            # Next action = place leather patch
            positions = np.argwhere(self[player].board==0)
            actions = []
            if len(positions) > 0:
                actions = [PatchworkAction(self.game.size,0,0,(int(p[0]),int(p[1]))) for p in positions ]
            
            # Add discard leather patch action
            actions.append(PatchworkAction(self.game.size,0,1,(0,0)))
#                print(actions)
            return actions
        
        # Get player board
        board = self[player].board

        actions = []

        for deckPos in range(min(3, len(self.patchDeck))):

#            print (f"deckPos: {deckPos}, deck: {self.patchDeck}")
            patch = self.game.patches[int(self.patchDeck[deckPos] - 1)]

            # Use the patch if there are enough buttons
            if patch[P_PRICE] <= self[player].buttons:
                patchFigure = np.array(patch[P_FIGURE])
                orientations = ORIENTATION_TYPES[patch[P_ORIENTATION_TYPE]]

#                print (f"pos: {deckPos}, patch: {patchFigure}, orientations: {orientations}:")
                # Review all possible combinations
                for j in orientations:
                    patchOption = self.getPatchOrientation(patchFigure, j)

#                    print (f"figure: {patchOption}, shape: {patchOption.shape}")
                    # Find positions
                    for x in range(self.game.size - patchOption.shape[0] + 1):
                        for y in range(self.game.size - patchOption.shape[1] + 1):
#                            print(f"Action: ({self.patchDeck[deckPos]},{j},{x},{y}))")

                            # Validate patch orientation
                            fits = np.all(1 - np.multiply(board[x:x+patchOption.shape[0], y:y+patchOption.shape[1]],patchOption))
#                            print (f"fits: {fits}")
                            if fits: actions.append(PatchworkAction(self.game.size,deckPos+1,j,(x,y)))

        # Add advance Action
        if self[player].pos <= self[1 - player].pos and self[player].pos < self.game.boardButtons[-1]:
            actions.append(PatchworkAction(self.game.size,0,7,(0,0)))

#        print (len(actions))
        return actions

    def isChanceState(self):
        # For Patchwork Express, if there are 5 patches left, add blue tokens randomly
        return self.game.version == 1 and len(self.patchDeck) <= 5

    def generateRandomInformationSet(self, player=None):
        # For Patchwork Express, if there are 5 patches left, add blue tokens randomly
        bluePatches = np.arange(len(self.game.patches) - BLUE_PATCHES_INIT_POS + 1) + BLUE_PATCHES_INIT_POS
        self.game.randomGenerator.shuffle(bluePatches)

        self.patchDeck = np.append(self.patchDeck,  bluePatches)

        setId = ""
        for patch in bluePatches:
            setId += str(patch-16)
        
        return setId

    def isTerminal(self):
        return self[0].pos >= self.game.boardButtons[-1] and self[1].pos >= self.game.boardButtons[-1]
    
    def getResult(self):
#        print (self[0].pos, self[1].pos)

        # Determine the number of buttons you have left , adding the value of the special tile (if available). From this score, subtract 2 points for each empty space of your quilt boar
        if self[0].pos >= self.game.boardButtons[-1] and self[1].pos >= self.game.boardButtons[-1]:
            value1 = self[0].buttons - len(np.argwhere(self[0].board==0)) * 2 + 7 if self.bonus7X7 == 0 else 0
            value2 = self[1].buttons - len(np.argwhere(self[1].board==0)) * 2 + 7 if self.bonus7X7 == 1 else 0
            if value1 > value2:
                return [1, -1]
            if value2 > value1:
                return [-1, 1]
            if self.lastPlayer == 0: # In case of a tie, the player who got to the final space of the time board first win
                return [-1, 1]
            else:
                return [1, -1]
        return [0, 0]


    def getPatchOrientation(self, patchFigure, orientation):
        newPatch = np.copy(patchFigure)
        if orientation >= 4:
            newPatch = np.flip(newPatch, axis = 0)
        return np.rot90(newPatch, orientation%4)
        
    def getNextPlayer(self):
        if self.lastPlayer is None:
            return self.initPlayer
        
        if (self.nextLeatherPatchIndex < len(self.game.leatherPatches)):
            nextLeatherPatch = self.game.leatherPatches[self.nextLeatherPatchIndex] 
            if self[0].pos >= nextLeatherPatch:
                return 0
            if self[1].pos >= nextLeatherPatch:
                return 1
        
        if (self[0].pos < self[1].pos):
            return 0
        if (self[1].pos < self[0].pos):
            return 1
        return self.lastPlayer  # if both positions are equal, last player repeat

    def playerRepr(self, player):
        return ["1", "2"][player]

    def __repr__(self):
        s = ""
        for x in range(self.game.size):
            for p in range(2):
                for y in range(self.game.size):
                    s += ".x"[int(self[p].board[x,y])] + " "
                if p == 0 and x == 1:
                    if (self[0].buttons >= 0): s+= " "
                    if (self[0].buttons > -10 and self[0].buttons < 10): s+= " "
                    s += " " + str(int(self[0].buttons)) + "  "
                elif p == 0 and x == 2:
                    if (self[0].value < 10): s+= " "
                    s += "  " + str(int(self[0].value)) + "  "
                elif p == 0 and x == 3 and self.bonus7X7 == 0:
                    s += " 7x7  "
                elif p == 0 and ((x == 5 and self.game.version == 0) or (x==4 and self.game.version == 1)):
                    if (self[1].buttons >= 0): s+= " "
                    if (self[1].buttons > -10 and self[1].buttons < 10): s+= " "
                    s += " " + str(int(self[1].buttons)) + "  "
                elif p == 0 and ((x == 6 and self.game.version == 0) or (x==5 and self.game.version == 1)):
                    if (self[1].value < 10): s+= " "
                    s += "  " + str(int(self[1].value)) + "  "
                elif p == 0 and x == 7 and self.bonus7X7 == 1:
                    s += " 7x7  "
                else:
                    s+= "      "
            s+= "\n"
        for i in self.patchDeck:
            if (i < 10): s+= " "
            s += str(i) + " "
        s+= "\n"

        leatherPatchIndex = self.nextLeatherPatchIndex
        boardButtonIndex = 0

        for i in range (1, self.game.boardButtons[-1] + 1):
            player = 0 if self.lastPlayer is None else self.lastPlayer
            if i == self[player].pos:
                s+= "12"[player]
            elif i == self[1 - player].pos:
                s+= "12"[1 - player]
            elif leatherPatchIndex < len(self.game.leatherPatches) and i == self.game.leatherPatches[leatherPatchIndex]:
                s+= chr(9632)
            elif i == self.game.boardButtons[boardButtonIndex]:
                s+= "o"
            else:
                s+="."

            if leatherPatchIndex < len(self.game.leatherPatches) and i == self.game.leatherPatches[leatherPatchIndex]:
                leatherPatchIndex += 1
            if i == self.game.boardButtons[boardButtonIndex]:
                boardButtonIndex += 1

        s += "\n"
        return s
    
    def getObservationTensor(self, player):
        n_extraInfo = np.zeros(6 + len(self.game.patches))

        # Main board

        # buttons
        n_extraInfo[0] = self[player].buttons / 100
        n_extraInfo[1] = self[1 - player].buttons / 100

        # Distance to take buttons
        n_extraInfo[2] = (0 if self[ player ].pos >= self.game.boardButtons[-1] else self.game.boardButtons[self[ player ].nextButtonIndex] - self[ player ].pos) / 10
        n_extraInfo[3] = (0 if self[1-player].pos >= self.game.boardButtons[-1] else self.game.boardButtons[self[1-player].nextButtonIndex] - self[1-player].pos) / 10

        # Distance to next 1x1 patch
        if self.nextLeatherPatchIndex < len(self.game.leatherPatches):
            nextLeatherPatch = self.game.leatherPatches[self.nextLeatherPatchIndex]
            n_extraInfo[4] = (0 if self[ player ].pos >= nextLeatherPatch else nextLeatherPatch - self[ player ].pos) / 10
            n_extraInfo[5] = (0 if self[1-player].pos >= nextLeatherPatch else nextLeatherPatch - self[1-player].pos) / 10

        # patchDeck
        deck = np.copy(self.patchDeck)
        deck.resize(len(self.game.patches))
        n_extraInfo[6:] = deck / len(self.game.patches)

        return (np.copy(self[player].board), n_extraInfo)


    def getObservationSymmetries(self, player, policyTuple):
        SYM_ORIENTATIONS = [
          [[0,0,0,0,0,0,0,0]],

          [[0,1,0,1,0,1,0,1],
           [1,0,1,0,1,0,1,0]],

          [[0,1,2,3,2,3,0,1],
           [1,2,3,0,1,2,3,0],
           [2,3,0,1,0,1,2,3],
           [3,0,1,2,3,0,1,2]],

          [[0,1,0,1,4,5,4,5],
           [1,0,1,0,5,4,5,4],
           [],
           [],
           [4,5,4,5,0,1,0,1],
           [5,4,5,4,1,0,1,0]],

          [[0,1,2,3,4,5,6,7],
           [1,2,3,0,7,4,5,6],
           [2,3,0,1,6,7,4,5],
           [3,0,1,2,5,6,7,4],
           [4,5,6,7,0,1,2,3],
           [5,6,7,4,3,0,1,2],
           [6,7,4,5,2,3,0,1],
           [7,4,5,6,1,2,3,0]],

          [[0,0,0,0,0,0,0,0],
           [1,1,1,1,1,1,1,1],
           [2,2,2,2,2,2,2,2],
           [3,3,3,3,3,3,3,3],
           [4,4,4,4,4,4,4,4],
           [5,5,5,5,5,5,5,5],
           [6,6,6,6,6,6,6,6],
           [7,7,7,7,7,7,7,7]],

      ]

        # There are symmetries only in the player board. some options need to be adjusted to the board symmetry

        policyCube = np.reshape(policyTuple[0], (PATCH_ACTIONS + 1, ORIENTATIONS, self.game.size, self.game.size))
        totalPatches = min(PATCH_ACTIONS, len(self.patchDeck))  + 1 # Plus 1x1 Patch

#        print(policyCube)

        observationTensor = self.getObservationTensor(player)
        board0 = np.copy(observationTensor[0])
        extraInfo = np.copy(observationTensor[1])

        symmetries = []
        for symOrientation in range(ORIENTATIONS):
            # Calculate symmetry observation
            newObservation = (self.getPatchOrientation(board0, symOrientation), extraInfo)

            # Calculate new policies
            newPolicy = np.copy(policyCube)
            # Update each patch action. less Advance Action
            for patchPos in range(totalPatches): 
                if patchPos == 0:
                    patchNumber = 0
                    orientationType = 5
                else:
                    patchNumber = self.patchDeck[patchPos - 1]
                    patch=self.game.patches[patchNumber - 1]
                    orientationType = patch[P_ORIENTATION_TYPE]

                # switch action orientations based on valid patch's orientations
                for patchOrientation in ORIENTATION_TYPES[orientationType]:
#                    print (f"patchOrientation: {patchOrientation}, orientationType: {orientationType}")
                    newOrientation = SYM_ORIENTATIONS[orientationType][patchOrientation][symOrientation]
#                    print (f"newOrientation: {newOrientation}")

                    positions = policyCube[patchPos, patchOrientation] # Matrix of positions for patch and orientation actions

                    newPositions = self.getSymmetryPositions(positions, patchNumber, patchOrientation, symOrientation)

                    newPolicy[patchPos, newOrientation] = newPositions

            symmetries.append((newObservation, (np.ravel(newPolicy), )))

        action = PatchworkAction.fromStandardAction(self.game.size, np.argmax(newPolicy))

        if action.patchPos > 0:
            patchNumber = self.patchDeck[action.patchPos - 1]
            po = self.getPatchOrientation(np.array(self.game.patches[patchNumber - 1][P_FIGURE]), action.orientation)
            if action.position[0] + po.shape[0] <= self.game.size and action.position[1] + po.shape[1] <= self.game.size:
                pass
            else:
#                    print(newBoard)
#                    print (f"sym: {symOrientation}, p: {p}, o: {o}, pos: ({x},{y})")
                assert action.position[0] + po.shape[0] <= self.game.size and action.position[1] + po.shape[1] <= self.game.size
    
        return symmetries

    def getSymmetryPositions(self, positionPolicy, patchNumber, patchOrientation, newOrientation):
        if patchNumber == 0:
            h = 0
            v = 0
        else:
            patchFigure = np.array(self.game.patches[patchNumber - 1][P_FIGURE])
            h = patchFigure.shape[1 - patchOrientation%2] - 1 
            v = patchFigure.shape[patchOrientation%2] - 1

        orientationOption = newOrientation if newOrientation < 4 else 7 - newOrientation

        if orientationOption == 0:
            shift = (0,0)
        elif orientationOption == 1:
            shift = (-h, 0)
        elif orientationOption == 2:
            shift = (-v, -h)
        else:
            shift = (0,-v)

        if newOrientation < 4:
            axis = (0,1)
        else: 
            axis = (1,0)

#        print (f"getSymmetryPositions - positionPi: {positionPi}, newOrientation: {newOrientation}")
        return np.roll(self.getPatchOrientation(positionPolicy, newOrientation), shift, axis = axis)

    

    def validateOrientations(self):
        for patch in _PATCHES_FULL:
            visitedOrientations = []
            validOrientations = []
            for orientation in range(ORIENTATIONS):
                newPatch = self.getPatchOrientation(self, patch[P_FIGURE], orientation)
                if newPatch not in visitedOrientations:
                    validOrientations.append(orientation)
                    visitedOrientations.append(newPatch)
            assert patch[P_ORIENTATION_TYPE] == validOrientations
                
class PatchworkAlphaZeroModel():
    def __init__(self, size, totalPatches, dropout=0.3, learningRate=.001, numChannels = 128):
        self.size = size
        self.totalPatches = totalPatches
        self.dropout = dropout
        self.learningRate= learningRate
        self.numChannels = numChannels
        pass
    
    def initModel(self, weights=None):
#        from tensorflow import keras
        from keras.models import Model
        from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Input, Conv2D, Reshape, Concatenate
        from keras.optimizers import Adam


        # game params
        self.extraInfoSize = (6 + self.totalPatches, )
        self.action_size = (PATCH_ACTIONS + 1) * ORIENTATIONS * self.size*self.size

        # Neural Net
        #self.input_boards = Input(shape=(self.board_z, self.board_x, self.board_y))    # s: batch_size x board_x x board_y
        self.input_boards = Input(shape=(self.size,self.size))
        self.input_extra_info = Input(shape=self.extraInfoSize)    # s: batch_size x board_x x board_y

        input_boards_reshaped = Reshape((self.size, self.size, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization()(Conv2D(self.numChannels, 3, padding='same')(input_boards_reshaped)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization()(Conv2D(self.numChannels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization()(Conv2D(self.numChannels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization()(Conv2D(self.numChannels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        h_ef1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(16)(self.input_extra_info))))  # batch_size x 16
        h_ef2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(16)(h_ef1))))  # batch_size x 16
        h_concat = Concatenate()([h_conv4_flat, h_ef2])       
        s_fc1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256)(h_concat))))  # batch_size x 256
        s_fc2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256)(s_fc1))))          # batch_size x 256
        self.policy = Dense(self.action_size, activation='softmax', name='policy')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        model = Model(inputs=[self.input_boards, self.input_extra_info], outputs=[self.policy, self.v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.learningRate))

        return model


