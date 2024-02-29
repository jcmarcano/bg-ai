from bg_ai import Game, State, ComponentContainer 
import numpy as np

#Constants
DETECTIVE = 0
JACK = 1

class MrJackGame(Game):
    def initGame(self, characters=8):
        self.characters = characters

    def getAgentDefaults(self, agentName):
        if agentName == "MCTSAgent":
            return {
                "iterMax": 5000
            }
        return super().getAgentDefaults()

    def getModel(self):
        from keras.models import Model
        from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input

        self.dropout = 0.3

        # Neural Net
        observations = Input(shape=self.getObservationShapes())    

        s_fc1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(32)(observations))))  # batch_size x 128
        s_fc2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(32)(s_fc1))))  # batch_size x 128
        s_fc3 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(32)(s_fc2))))  # batch_size x 128
        policy = Dense(self.getActionSize()[0], activation='softmax', name='pi0')(s_fc3)   # batch_size x market actions
        value = Dense(1, activation='tanh', name='v')(s_fc3)                    # batch_size x 1

        return Model(inputs=observations, outputs=[policy, value])

    def getObservationShapes(self):
        return (32,)
    
    def getActionSize(self):
        return (16,)
    
    def playerRepr(self, player):
        return ["Detective", "Jack"][player]



class MrJackState(State):
    def initState(self):
        if self.initPlayer != 0:
            raise Exception("This game does not allow to change initial Player")

        ## Public Components
        self.addComponent("round",    initValue=1)
        self.addComponent("witness",  initValue=0) # Invisible
        self.addComponent("visible",  initValue=np.array([0] * self.game.characters))
        self.addComponent("innocent", initValue=np.zeros(self.game.characters))
        self.addComponent("cards",    initValue=np.array([]))

        self.shuffleCards()


        ## Player Private Components (Only Jack knows this info)
        self.addComponent("jack", ComponentContainer.PRIVATE_ACCESS, isPlayerComponent=True)
        self[DETECTIVE].jack = "?"
        self[JACK].jack = self.game.randomGenerator.choice(range(1, self.game.characters + 1))


    def getValidActions(self, player):
        initPos = 0 if len(self.cards) <= 4 else 4

        # return every available card in the subgroup
        # oup of 4 cards
        actions = [i*self.game.characters + int(self.cards[initPos + j] - 1) for i in range(0,2) for j in range(len(self.cards) - initPos)]
        return actions

    def performAction(self, action, player):
        character = action % self.game.characters + 1
        visible = action // self.game.characters

        assert character in range(1, self.game.characters + 1) and visible in [0,1]

        self.cards = np.delete(self.cards, np.where(self.cards == character))
        self.visible[character - 1] = visible


        if (len(self.cards)%4 == 0):
            self.witness = self.visible[self[JACK].jack - 1]
            self.innocent = np.array([self.innocent[i] or int(self.visible[i] != self.witness) for i in range(self.game.characters)])
            self.round += 1

        self.lastPlayer = player

    def getNextPlayer(self):
        if self.lastPlayer is None:
            return DETECTIVE
        
        if len(self.cards) % 4 == 2:
            return self.lastPlayer
        else:
            return 1 - self.lastPlayer
    
    def isChanceState(self):
        return len(self.cards) == 0 

    def generateRandomInformationSet(self, player=None):
        setId = ""
        if self.hasPrivateComponents():
            assert player is not None
            # Fill hidden information for other player
            if player == JACK:
                self[DETECTIVE].jack = "-" # Not need a real value, but have to fill the private component to complete the state
                setId="0"
            else:
                # Build an Information Set getting Jack from the list of suspects (non-innocents)
                self[JACK].jack = self.game.randomGenerator.choice([character for character in range(1,self.game.characters+1) if self.innocent[character - 1] == 0]) 
                setId = self[JACK].jack

        if self.isChanceState():
            # InformationSet for Chance State
            
            # Shuffle cards
            self.shuffleCards()
            for card in self.cards[0:4]:
                setId += str(card)

        return setId
        
    def isTerminal(self):
        return self.getResult() != [0,0]
    
    def getResult(self):
        # Jack wins if Detective can not catch Jack in 8 turns
        if (self.round == 9):
            return [-1, 1]
        # Detective wins if there is just one character "Not Innocent" and there is at least one move to catch him
        if (np.count_nonzero(self.innocent) == self.game.characters - 1) and (self.round < 8 or len(self.cards) > 1):
            return [1, -1]
        return [0, 0]

    def actionRepr(self, action, player):
        return f"{action%self.game.characters+1} {['Invisible', 'Visible'][action//self.game.characters]}"
    
    def __repr__(self):
        s = "J: " + (str(self[self.observationPlayer].jack if self.isObservation else self[JACK].jack)) + "         " + ("Visible" if self.witness else " Hidden") + "\n"
        initPos = 0 if len(self.cards) <= 4 else 4
        s += "R: " + str(self.round) + "  "*((4- len(self.cards))%4) + "    C: " + np.array2string(self.cards[initPos: initPos + 4]) + "\n"
        s += "V: " + np.array2string(self.visible) + "\n"
        s += "I: " + np.array2string(self.innocent) + "\n"
        return s

    def shuffleCards(self):
        deck = np.arange(1, self.game.characters + 1)
        self.game.randomGenerator.shuffle(deck)
        self.cards = np.concatenate((np.sort(deck[:4]),np.sort(deck[4:])))

