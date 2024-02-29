import sys
from bg_ai import Agent

class HumanAgent(Agent):
    def initAgent(self, actionsAsOptions = False, hideOptions = False):
        self.actionsAsOptions = actionsAsOptions
        self.hideOptions = hideOptions
    
    def getActionKey(self, action):
        if type(action) == int:
            return int(action)
        if type(action) == tuple:
            return [int(subAction) for subAction in action]
        
    def selectAction(self, observation):
        actions = observation.getValidActions(self.player)
        print(actions)
        if (len(actions)) == 0:
            raise Exception ("No available actions")
        actions.sort(key=self.getActionKey)
        if not self.hideOptions:
            for i, action in enumerate(actions):
                if self.actionsAsOptions:
                    print(observation.actionRepr(action, self.player))
                else:
                    print(f"{i + 1}: {observation.actionRepr(action, self.player)}")
        print(f"Player {self.game.playerRepr(self.player)}. Select an action")
        while True:
            try:
                selectedOption = int(input()) - 1
                if self.actionsAsOptions:
                    if selectedOption in actions:
                        return selectedOption
                else:
                    return actions[selectedOption]
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                pass

            if self.actionsAsOptions:
                if (self.hideOptions):
                    print (f"Invalid Action. Allowed Actions: {[observation.actionRepr(action) for action in actions]}")
                else:
                    print ("Invalid Action")
            else:
                print (f"Action must be an integer between {1} and {len(actions)}")

        