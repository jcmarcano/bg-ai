from bg_ai import Game, State, ComponentContainer, LogManager
import numpy as np
import math
import copy
import itertools
from providers.keras import KerasProvider


TAKE = 0
SELL = 1

## Goods
CAMEL = 0
LEATHER = 1
SPICE = 2
CLOTH = 3
SILVER = 4
GOLD = 5
DIAMONDS = 6


## Good Tokens distribution
# Sort the tokens by goods type.
# Make a pile for each goods type in order of value

GOOD_TOKEN_VALUES = [[]]*7
GOOD_TOKEN_VALUES[LEATHER]  = [1, 1, 1, 1, 1, 1, 2, 3, 4]
GOOD_TOKEN_VALUES[SPICE]    = [1, 1, 2, 2, 3, 3, 5]
GOOD_TOKEN_VALUES[CLOTH]    = [1, 1, 2, 2, 3, 3, 5]
GOOD_TOKEN_VALUES[SILVER]   = [5, 5, 5, 5, 5]
GOOD_TOKEN_VALUES[GOLD]     = [5, 5, 5, 6, 6]
GOOD_TOKEN_VALUES[DIAMONDS] = [5, 5, 5, 7, 7]

GOOD_CARDS = [0]* 7
GOOD_CARDS[CAMEL]    = 8
GOOD_CARDS[LEATHER]  = 10
GOOD_CARDS[SPICE]    = 8
GOOD_CARDS[CLOTH]    = 8
GOOD_CARDS[SILVER]   = 6
GOOD_CARDS[GOLD]     = 6
GOOD_CARDS[DIAMONDS] = 6

BONUS_TOKENS= [[]]*3
BONUS_TOKENS[0] = [1, 1, 2, 2, 2, 3, 3]
BONUS_TOKENS[1] = [4, 4, 5, 5, 6, 6]
BONUS_TOKENS[2] = [8, 8, 9, 10, 10]

## Bonus expected value
BONUS_EV = [2, 5, 9]

MAX_ALLOWED_EXCHANGES = 10

OBSERVATION_SHAPE = (28,) 

logger = LogManager().getLogger(__name__)



class JaipurGame(Game):
    def getAgentDefaults(self, agentName):
        if agentName == "MCTSAgent":
            return {
                "iterMax": 1000
            }
        if agentName == "AlphaZeroAgent":
            return {
                "iterMax": 100,
                "model": JaipurAlphaZeroModel(),
                "provider": KerasProvider,
            }
        return super().getAgentDefaults()
    
    def getObservationShapes(self):
        return (OBSERVATION_SHAPE,)
    
    def getActionSize(self):
        return (JaipurAction.MARKET_ACTIONS,JaipurAction.PLAYER_ACTIONS)
    
    def getModel(self):
        return JaipurAlphaZeroModel()


class JaipurAction():
    ## Action Types
    MARKET = 0
    PLAYER = 1

    MARKET_ACTIONS = 32
    # Action Type           Actions     Values                  Combinations (Market cards, action Cards)
    # Sell                     1        ()                      1
    # Take Goods               5        (1,) .. (6,)            (5,1)
    # Exchange (Market)       26        (1,2) .. (1,2,3,4,5)    (5,2) + (5,3) + (5,4) + (5,5) [10 + 10 + 5 +1]
    # Take camels included in actions when all gods are camels

    PLAYER_ACTIONS = 128
    # Action Type           Actions     Values                  Combinations (Hand cards, action Cards)
    # Sell/Exchange (Hand)   128        () .. (1,2,3,4,5,6,7)   (7,0) + (7,1) + (7,2) + (7,3) + (7,4) + (7,5) + (7,6) + (7,7) [1 + 7 + 21 + 35 + 35 + 21 + 7 + 1]
        
    _ACTION_PORTIONS = [(5, [1, 6, 16, 26, 31, 32], MARKET_ACTIONS), 
                       (7, [1, 8, 29, 64, 99, 120, 127, 128], PLAYER_ACTIONS)]

    def __init__(self, subAction, cards):
        assert subAction in (self.MARKET, self.PLAYER)
        self.subAction = subAction
        self.cards = cards

    def __int__(self):
        return JaipurAction.getActionValue(self.cards, self._ACTION_PORTIONS[self.subAction])

    @staticmethod
    def fromStandardAction(subAction, action):
        cards = JaipurAction.getCardsCombination(action, JaipurAction._ACTION_PORTIONS[subAction])
        return JaipurAction(subAction, cards)


    @staticmethod
    def getCardsCombination(action, option):
        initPos = 0
        for pos, value in enumerate(option[1]):
            if action < value:
                options = pos
                break
            else:
                initPos = action - value
        move = ()
        if (options > 0):
            for pos, combination in enumerate(itertools.combinations(range(1, option[0] + 1), options)):
                if pos == initPos:
                    move = combination
                    break
        
        return move

    @staticmethod
    def getActionValue(cards, option):
        if cards == (): return 0
        options = len(cards)
        initPos = option[1][options - 1]

        action = 0
        for pos, combination in enumerate(itertools.combinations(range(1, option[0] + 1), options)):
            if cards == combination:
                action = initPos + pos
                break

        return action

    def __repr__(self):
        return repr(self.cards)

    def __eq__(self, other):
        return self.cards == other.cards


class JaipurState(State):
    def initState(self): 
        ## Public Components
        self.addComponent("cardsInDeck",      initValue=sum(GOOD_CARDS))
        self.addComponent("goodTokens",       initValue=[len(tokens) for tokens in GOOD_TOKEN_VALUES])
        self.addComponent("totalBonusTokens", initValue=[len(tokens) for tokens in BONUS_TOKENS])
        self.addComponent("market",           initValue=[CAMEL, CAMEL, CAMEL])   # Place 3 camel cards face up between the players
        self.addComponent("exchangeCount",    initValue=0)   # Place 3 camel cards face up between the players

        ## Private Components
        self.addComponent("goodDeck",    ComponentContainer.PRIVATE_ACCESS, initValue=copy.deepcopy(GOOD_CARDS))
        self.addComponent("bonusTokens", ComponentContainer.PRIVATE_ACCESS, initValue=copy.deepcopy(BONUS_TOKENS))

        ## Player Public Components
        self.addComponent("cardsInHand", initValue=0, isPlayerComponent=True)
        self.addComponent("bonusTokensInHand", initValue=0, isPlayerComponent=True)
        self.addComponent("goodTokensInHand",  initValue=0, isPlayerComponent=True)

        ## Player Private Components
        self.addComponent("hand",        ComponentContainer.PRIVATE_ACCESS, initValue=[], isPlayerComponent=True)
        self.addComponent("herd",        ComponentContainer.PRIVATE_ACCESS, initValue=0,  isPlayerComponent=True)
        self.addComponent("points",      ComponentContainer.PRIVATE_ACCESS, initValue=0,  isPlayerComponent=True)


        ## To reduce randomness, this game infer private information based on public information. This inferred information is public to maintain it along the game and is stored as player components
        self.addComponent("inferred_goodDeck", initValue=copy.deepcopy(GOOD_CARDS), isPlayerComponent=True)
        self.addComponent("inferred_hand",      initValue=[],                        isPlayerComponent=True)
        self.addComponent("inferred_herd",      initValue=0,                         isPlayerComponent=True)
        self.addComponent("inferred_points",    initValue=0,                         isPlayerComponent=True)


        # Init Setup

        for player in range(2):
            # Deal 5 cards to each player.
            self.selectCards(self[player].hand, numCards=5, player=player)

            # The players then remove any camels from their hands and put them face up in a in a stack in front of them. This forms each player’s herd.
            if CAMEL in self[player].hand: 
                camelCount = self[player].hand.count(CAMEL)
                self[player].herd += camelCount
                self[player].hand = self[player].hand[camelCount:]

                # Update inferred info
                self[1-player].inferred_goodDeck[CAMEL] -= camelCount
                self[1-player].inferred_herd += camelCount
            self[player].cardsInHand = len(self[player].hand)

        # Take the first two cards from the deck and place them face up next to the camels. (There may well be 1 or 2 camels drawn.) Te market is now ready
        self.selectCards(self.market, numCards=2) 

        assert self.cardsInDeck == sum(self.goodDeck)
        assert sum(self[0].inferred_goodDeck) == self.cardsInDeck + self[1].cardsInHand
        assert sum(self[1].inferred_goodDeck) == self.cardsInDeck + self[0].cardsInHand

        # Shuffle Bonus Tokens
        for level in range(3):
            self.game.randomGenerator.shuffle(self.bonusTokens[level])

    def performAction(self, action, player):
        playerHand = self[player].hand
        playerHerd = self[player].herd
        playerPoints = self[player].points
        inferredHand = self[1-player].inferred_hand
        inferredHerd = self[1-player].inferred_herd
        inferredPoints = self[1-player].inferred_points

        # print(f"action: {action}, market: {self.market}, hand: {playerHand}, herd: {playerHerd}")
        assert len(action) == 2
        assert isinstance(action[0], JaipurAction) and isinstance(action[1], JaipurAction)

        marketCards = action[0].cards
        playerCards = action[1].cards
        turnAction = SELL if marketCards == () else TAKE
        assert turnAction >= TAKE and turnAction <= SELL


        if (turnAction == TAKE):
            marketGoods = [self.market[card - 1] for card in marketCards]
            playerGoods = [playerHand[card - 1] for card in playerCards]
            camelCount = marketGoods.count(CAMEL)

#            print(f"market Goods: {marketGoods}, player Goods: {playerGoods}, camelCount: {camelCount}")
            assert (camelCount == 0 and ((len(marketCards) == 1 and len(playerCards) == 0) or (len(marketCards) > 1 and len(marketCards) >= len(playerCards)))) or camelCount == self.market.count(CAMEL)
            assert len(marketCards) <= len(self.market) and (len(playerHand) + (len(marketCards) - marketGoods.count(CAMEL)) - len(playerCards)) <= 7
            assert len(marketCards) < 2 or (len(marketCards) - camelCount - len(playerCards)) <= playerHerd

            for i, good in enumerate(marketGoods):
                self.market.remove(good)
                if (good == CAMEL):  # Take good action
                    playerHerd += 1
                    inferredHerd += 1
                else:
                    playerHand.append(good)
                    inferredHand.append(good)
        
                if len(marketCards) >=2 and camelCount == 0: # If there are more than two cards from market and none of them are camels, this is an exchange action
                    if i < len(playerGoods):
                        playerGood = playerGoods[i]
#                        print(f"player Good: {playerGood}")
                        playerHand.remove(playerGood)
                        # If tha card was inferred, remove it from the hand. otherwise, remove it from the deck
                        if playerGood in inferredHand:
                            inferredHand.remove(playerGood)
                        else: 
                            self[1-player].inferred_goodDeck[playerGood] -= 1
                        self.market.append(playerGood)
                    else: ## remaining exchange cards are camels
#                        print(f"good is a camel")
                        assert playerHerd > 0
                        playerHerd -= 1
                        inferredHerd -= 1
                        self.market.append(CAMEL)
                    self.exchangeCount +=1
                else:
                    self.exchangeCount = 0

        elif (turnAction == SELL):
            goods = [playerHand[card - 1] for card in playerCards]
            assert len(playerCards) <= len(playerHand) and len(goods) > 0
            goodCount = len(goods)
            lastGood = goods[0]
            for good in goods:
                if (good != lastGood):
                    print("ERROR good != lastGood")
                    print(self)
                    print (action)
                    print (player)
                assert good == lastGood
                playerHand.remove(good)
                # If tha card was inferred, remove it from the hand. otherwise, remove it from the deck
                if good in inferredHand:
                    inferredHand.remove(good)
                else: 
                    self[1-player].inferred_goodDeck[good] -= 1
                if (self.goodTokens[good] > 0):
                    self.goodTokens[good] -= 1
                    playerPoints += GOOD_TOKEN_VALUES[good][self.goodTokens[good]]
                    inferredPoints += GOOD_TOKEN_VALUES[good][self.goodTokens[good]]
                    self[player].goodTokensInHand += 1
            
            if goodCount >= 3:
                # If you sell 3 or more cards, take the corresponding bonus token (if any).
                level = min(goodCount - 3, 2)
                if self.totalBonusTokens[level] > 0:
                    bonus = self.bonusTokens[level].pop()
                    self.totalBonusTokens[level] = len(self.bonusTokens[level])

                    playerPoints += bonus
                    self[player].bonusTokensInHand += 1

                    # Use expected value to infer bonus
                    inferredPoints += BONUS_EV[level]
            self.exchangeCount = 0


        self.market.sort()
        playerHand.sort()
        inferredHand.sort()

        self[player].hand = playerHand
        self[player].cardsInHand = len(playerHand)
        self[player].herd = playerHerd
        self[player].points = playerPoints

        self[1-player].inferred_hand = inferredHand
        self[1-player].inferred_herd = inferredHerd
        self[1-player].inferred_points = inferredPoints

        self.lastPlayer = player

##        print(f"after action, market: {self.market}, hand: {playerHand}, herd: {playerHerd}, points: {playerPoints}")

    def isTerminal(self):
        ## END OF A ROUND
        # A round ends immediately if:
        # - 3 types of goods token are depleted.
        # - There are no cards left in the draw pile when trying to fill the market. => market has less than 5 cards

        return self.goodTokens.count(0) >= 4 or (self.cardsInDeck == 0 and len(self.market) < 5) 


    def getValidActions(self, player):
        """ Get all possible actions from this state.
            Each action is a combination of the market cards and player cards used in the action
            Action = (Action Type, tuple with selected market cards, tuple with selected player cards)
        """


        # On your turn, you can either:
        # - Take Cards 
        # - Sale Cards
        # But never both!

        # Current player Hand and Herd
        playerHand = self[player].hand
        playerHerd = self[player].herd

        logger.debug(f"Market: {self.market}, Hand: {playerHand}, Herd: {playerHerd}")
        actions = []

        camelCount = self.market.count(CAMEL)

        ##### TAKE CARDS #####
        # If you take cards, you must choose one of the following options:
        # A take several goods (=EXCHANGE !),
        # B take 1 single good, 
        # C take all the camels.

        # A. Take several goods (Exchange)
        if (camelCount <= 3 and self.exchangeCount < MAX_ALLOWED_EXCHANGES):
            cards = self.market[camelCount:]
            ## Take all the goods cards that you want into your hand (they can be of different types)
            for i in range(2, len(cards) + 1):
                actionOptions = set()
                for cardSet in itertools.combinations(cards, i): # Get all combination of at least 2 cards in the market
                    # Create possible exchange options
                    actionOptions.add(cardSet)
                for actionOption in actionOptions:
                    validOptions = [CAMEL] * playerHerd + [card for card in playerHand if card not in actionOption] # Complete the set with camels
                    exchangeOptions = set()

                    # then ... exchange the same number of cards. Te returned cards can be camels, goods, or a combination of the two.
                    for exchangeSet in itertools.combinations(validOptions, i): ## Get all unique combinations of the same size of the market set (i)
                        exchangeOptions.add(exchangeSet)
                    for exchangeOption in exchangeOptions:
                        camelCountInSet = exchangeOption.count(CAMEL)
                        #print(f"playerHand: {playerHand}, actionOption: {actionOption}, exchangeOption{exchangeOption}, camelCountInSet: {camelCountInSet}")
                        if len(playerHand) + len(actionOption) - (len(exchangeOption) - camelCountInSet) <= 7: #Players may never have more than 7 cards in their hand at the end of their turn

                            # get the position of the selected cards in the market
                            pos = 0
                            marketCards = []
                            for good in actionOption:
                                pos = self.market.index(good, pos) + 1
                                marketCards.append(pos)
                            pos = 0

                            # get the position of the selected cards in the player hand
                            playerCards = []
                            for good in exchangeOption:
                                if good != CAMEL:
                                    pos = playerHand.index(good, pos) + 1
                                    playerCards.append(pos)
        
                            #print (f"marketCards: {marketCards}, playerCards: {playerCards}")

                            actions.append((JaipurAction(JaipurAction.MARKET,tuple(marketCards)), JaipurAction(JaipurAction.PLAYER,tuple(playerCards))))

        # Take 1 single Good
        if (camelCount < 5 and len(playerHand) < 7): 
            # Take a single goods card from the market into your hand
            actions += [(JaipurAction(JaipurAction.MARKET,(i + 1,)), JaipurAction(JaipurAction.PLAYER,())) for i in range(5) if self.market[i] != CAMEL and (i == 1 or self.market[i-1] != self.market[i])] 

        # Take Camels
        if (camelCount > 0):
            # Take ALL the camels from the market and add them to your herd
            camelCards = tuple([i + 1 for i, card in enumerate(self.market) if card == CAMEL])
#            print(f"camelCards: {camelCards}")
            actions.append((JaipurAction(JaipurAction.MARKET,camelCards), JaipurAction(JaipurAction.PLAYER,())))

        ##### SELL CARDS #####
        if (len(playerHand) > 0):
            lastGood = 0
            cards = []
            for i, good in enumerate(playerHand):
                if good != lastGood:
                    if ((lastGood < SILVER and len(cards) >= 1) or (lastGood >= SILVER and len(cards) >= 2)): # When selling the 3 most expensive goods (diamonds, gold and silver), the sale must include a minimum of 2 cards
                        actions.append((JaipurAction(JaipurAction.MARKET,()), JaipurAction(JaipurAction.PLAYER,tuple(cards))))
                    cards = []
                cards.append(i+1)
                lastGood = good

            if (lastGood < SILVER and len(cards) >= 1) or (lastGood >= SILVER and len(cards) >= 2): # When selling the 3 most expensive goods (diamonds, gold and silver), the sale must include a minimum of 2 cards
                actions.append((JaipurAction(JaipurAction.MARKET,()), JaipurAction(JaipurAction.PLAYER,tuple(cards))))


        logger.debug(f"actions: {actions}")  
        return actions
    
    def isChanceState(self):
        return len(self.market) < 5 and self.cardsInDeck > 0
    
    def generateRandomInformationSet(self, player=None):
        if self.isChanceState():
            # Generate the Information set for the chance state
            setId = self.selectCards(self.market, min(5 - len(self.market), self.cardsInDeck))
            return setId
        
        if self.hasPrivateComponents():
            # Use inferred info to generate the Information set to fill opponent Private information
            setId = "0" # Default value if no random information is generated
            inferredCards = copy.deepcopy(self[player].inferred_goodDeck)

            # Set other player hand as inferred hand plus random cards to complete current hand size
            self[1-player].hand = copy.deepcopy(self[player].inferred_hand)
            if len(self[player].inferred_hand) < self[1-player].cardsInHand:
                # complete hand with random cards. only this part of the generated information is random
                setId = self.selectCards(self[1-player].hand, self[1-player].cardsInHand - len(self[player].inferred_hand), deck=inferredCards)
            # Set deck after taking cards for the hand
            self.goodDeck = inferredCards

            # Inferred herd and points
            self[1-player].herd = self[player].inferred_herd
            self[1-player].points = self[player].inferred_points

            # Fill bonus tokens with expected values
            self.bonusTokens = [[BONUS_EV[level]]*total for level, total in enumerate(self.totalBonusTokens)]

            assert sum(inferredCards) == self.cardsInDeck
            assert len(self[1-player].hand) == self[1-player].cardsInHand

            for i in range(3):
                assert len(self.bonusTokens[i]) == self.totalBonusTokens[i]


            return setId


    def selectCards(self, target, numCards = 1, player=None, deck=None):

        assert deck is not None or self.cardsInDeck == sum(self.goodDeck)

        goodDeck = self.goodDeck if deck is None else deck
        assert numCards <= sum(goodDeck)

        selectedCards = []
        for _ in range(numCards):
            # Do not take Camels if getting card for inference Deck
            initCard = 0 if deck is None else deck[CAMEL] 
            card = self.game.randomGenerator.choice(np.arange(initCard, sum(goodDeck)))
            # print(f"random card {card}" )

            pos = 0
            selectedCard = -1
            for good in range(7):
                pos += goodDeck[good]
                if (card < pos):
                    selectedCard = good
                    goodDeck[good] -= 1
                    break
            assert selectedCard >= CAMEL

            target.append(selectedCard)

            # Do not infer good deck if card is taken from the inferred deck
            if deck is None:
                if player is None:
                    self[0].inferred_goodDeck[selectedCard] -= 1
                    self[1].inferred_goodDeck[selectedCard] -= 1
                else:
                    self[player].inferred_goodDeck[selectedCard] -= 1
            selectedCards.append(selectedCard)
   
        target.sort()
        if deck is None:
            self.cardsInDeck -= numCards
        selectedCards.sort()
        setId = ""
        for card in selectedCards:
            setId += str(card)

        assert len(setId) == numCards
        return setId

    def getResult(self):
        
        # The player with the most camels in his herd receives the camel token, worth 5 rupees.
        if self[0].herd > self[1].herd:
            self[0].points += 5
        if self[1].herd > self[0].herd:
            self[1].points += 5


        # The players turn over their tokens and add them up to determine who is the richer.

        #Option 1 value = math.tanh((self[0] - self[1])/3) # dividing by 3 takes a better distribution for tanh function

        # if self[0].points > self[1].points:
        #     return [value, -value]
        # if self[1].points > self[0].points:
        #     return [-value, value]
        

        if self[0].points != self[1].points:
            maxPoints = max(self[0].points, self[0].points)
            maxPoints = 100
            return [self[0].points/maxPoints, self[1].points/maxPoints]

        # In the case of a tie, the player with the most bonus tokens takes the seal.
        # if self[0].bonusTokensInHand > self[1].bonusTokensInHand: #returns a low value
        #     return [.1, -.1]
        # if self[1].bonusTokensInHand > self[0].bonusTokensInHand:
        #     return [-.1, .1]
        if self[0].bonusTokensInHand > self[1].bonusTokensInHand: #returns a low value
            return [1., .99]
        if self[1].bonusTokensInHand > self[0].bonusTokensInHand:
            return [.99, 1.]

        # If the players are still tied, the one with the most goods tokens takes the seal.
        # if self[0].goodTokensInHand > self[1].goodTokensInHand:
        #     return [.1, -.1]
        # if self[1].goodTokensInHand > self[0].goodTokensInHand:
        #     return [-.1, .1]
        if self[0].goodTokensInHand > self[1].goodTokensInHand:
            return [1., .99]
        if self[1].goodTokensInHand > self[0].goodTokensInHand:
            return [.99, 1.]
        
        return [0., 0.]
        
    def getObservationTensor(self, player):

        baseTensor = np.zeros(OBSERVATION_SHAPE[0])

        #market 
        market = np.array(self.market[0:5])
        market.resize(5)
        baseTensor[0:5] = market / 6 # Max good value

        #goods
        baseTensor[5:11] = np.array(self.goodTokens[1:]) / np.array([9,7,7,5,5,5])  # tokens on each stack

        #player cards
        hand0 = np.array(self[player].hand)
        hand0.resize(7)
        baseTensor[11:18] = hand0 / 6  # Max good value

        #player herd
        baseTensor[18] = self[player].herd / 11   # Total herds

        #opponent cards
        hand1 = np.array(self[player].inferred_hand)
        hand1.resize(7)
        baseTensor[19:26] = hand1 / 6   # Total herds

        #opponent herd
        baseTensor[26] = self[player].inferred_herd / 11   # Total herds

        #deck size
        baseTensor[27] = self.cardsInDeck / 40

        return (baseTensor,)

    def getObservationSymmetries(self, player, policyTuple):
        return [(self.getObservationTensor(player), policyTuple)]

    def actionRepr(self, action, player):
        text = ["camel", "leather", "cloth", "spice", "silver", "gold", "diamonds"]
        marketCards = [text[self.market[card - 1]] for card in action[0].cards]
        playerCards = [text[self[player].hand[card - 1]] for card in action[1].cards]

        if marketCards == []:
           return f"SELL {len(playerCards)} {playerCards[0]}"
        if marketCards[0] == 'camel':
            return f"TAKE {len(action[0].cards)} camel{'s' if len(marketCards) > 1 else ''}"
        if len(marketCards) == 1:
            return f"TAKE 1 {marketCards[0]}"
        else: 
            cards = f"{', '.join(playerCards)}" if len(playerCards) > 0 else ""
            camel = f"{len(marketCards) - len(playerCards)} camel" if len(marketCards) > len(playerCards) else ""
            if len(marketCards) - len(playerCards) > 1: camel += "s"
            conn = " and " if cards != "" and camel != "" else ""
            return f"EXCHANGE {', '.join(marketCards)} WITH " + cards + conn + camel

    def __repr__(self):
        text = ["CAME", "LEAT", "CLOT", "SPIC", "SILV", "GOLD", "DIAM"]
        s= ""
        for i in range(6):
            s += "MARKET"[i] + "  |  " + text[6-i] + ": "
            if (self.goodTokens[6 - i]) > 0:
                s += str(GOOD_TOKEN_VALUES[6 - i][self.goodTokens[6 - i] - 1]) + " ("  + str(self.goodTokens[6 - i]) + ")"
            else: 
                s += "-----"
            if i == 1:
                s+= "                      " + str(self.cardsInDeck)
            if i == 3:
                s += "     "
                for j in range(len(self.market)):
                    s += text[self.market[j]] + "  "
            s += "\n"

        for p in range(2):
            s += "---+--------------------------------------------\n"
            if not self.isObservation or self.observationPlayer == p:
                s += "P  |  PTS: "
                if self[p].points < 10: s+= " "
                s += str(self[p].points) + " (" + str(self[p].bonusTokensInHand + self[p].goodTokensInHand) + ")\n"
                s += str(p) + "  |  CAME: " + str(self[p].herd) + "   "
                for i in range(len(self[p].hand)):
                    s += text[self[p].hand[i]] + "  "
            else: 
                s += "P  |  PTS: "
                if self[1-p].inferred_points < 10: s+= " "
                s += str(self[1-p].inferred_points) + " (" + str(self[p].bonusTokensInHand + self[p].goodTokensInHand) + ")             (Inferred)\n"
                s += str(p) + "  |  CAME: " + str(self[1-p].inferred_herd) + "   "
                for i in range(len(self[1-p].inferred_hand)):
                    s += text[self[1-p].inferred_hand[i]] + "  "
                for j in range(len(self[1-p].inferred_hand), self[p].cardsInHand):
                    s += "????  "
            s += "\n"

        return s
    
class JaipurAlphaZeroModel():
    def __init__(self, dropout=0.3, learningRate=.001):
        self.dropout = dropout
        self.learningRate= learningRate
        pass
    
    def initModel(self, weights=None):
        from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
        from keras.models import Model
        from keras.optimizers import Adam


        # Neural Net
        self.observations = Input(shape=OBSERVATION_SHAPE)    

        s_fc1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128)(self.observations))))  # batch_size x 128
        s_fc2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128)(s_fc1))))  # batch_size x 128
        s_fc3 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128)(s_fc2))))  # batch_size x 128
        self.marketPolicy = Dense(JaipurAction.MARKET_ACTIONS, activation='softmax', name='policy0')(s_fc3)   # batch_size x market actions
        self.playerPolicy = Dense(JaipurAction.PLAYER_ACTIONS, activation='softmax', name='policy1')(s_fc3)   # batch_size x player actions
        self.value = Dense(1, activation='tanh', name='value')(s_fc3)                    # batch_size x 1
        model = Model(inputs=[self.observations], outputs=[self.marketPolicy, self.playerPolicy, self.value])
        model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.learningRate))
        if weights:
            model.set_weights(weights)
        return model

