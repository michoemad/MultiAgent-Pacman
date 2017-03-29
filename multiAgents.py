# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #print(gameState.hasFood(gameState
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        
        return legalMoves[bestIndices[0]]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        D  = 1000
        F = False
        #BASICALLY OBTAINS THE MINIMUM DISTANCE TO THE NEXT APPLE
        for i in range((newFood.width)):
            for j in range((newFood.height)):
                if (newFood[i][j]):
                    F= True
                    D = min(D,manhattanDistance((i,j),newPos))
        if (not F):
            D = 0
        G =0
        C =0
        #FOR CAPSULES
        for i in successorGameState.getCapsules():
            C += manhattanDistance(i,newPos)
        for i in newGhostStates:
            if (i.getPosition()):
                if (i.scaredTimer > 0 ):
                    G -= manhattanDistance(i.getPosition(),newPos)
                else:
                    G += 0.9*manhattanDistance(i.getPosition(),newPos)
        M=0
        for i in newScaredTimes:
            M += i
        #print successorGameState.getScore() - D + (0.9)*G + 20*M
        return  successorGameState.getScore() - (1.05)*D + (G) + 20*M -C

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def fmax(depth,state): #Returns (score,state), pacman only
            D= (-1)*float('inf')
            act = None
            if ((state.isWin())|(state.isLose())): #TERMINAL CONDITION MANZ
                return self.evaluationFunction(state)
            if (depth <=0):
                return state.getScore()
            for a in state.getLegalActions(0):
                s = state.generateSuccessor(0,a) # s is the new state
                score = fmin(s,1,depth) # caries score of what all agents would do given my action
                if (score>D):
                    D = score
                    act = a
            if (depth==self.depth): #IF i'm in the upper call
                #print D
                return act
            #print D
            return D
                #generate all possible moves
                #from each move see pac's response
                                    
        def fmin(state,index,depth):
            D = float('inf')
            if ((state.isWin())|(state.isLose())):
                return self.evaluationFunction(state)
            if ( index == (state.getNumAgents() -1)):
                #LAST AGENT
                for a in state.getLegalActions(index):
                    s = state.generateSuccessor(index,a)
                    D = min(D,fmax(depth-1,s)) # GO BACK TO PACMAN
            else:
                for a in state.getLegalActions(index):
                    s = state.generateSuccessor(index,a)
                    D = min(D,fmin(s,index+1,depth))
            return D
        return fmax(self.depth,gameState)

            
##        def all_poss(index,state):
##            L=[]
##            if (index >= state.getNumAgents()-1):
##                for a in state.getLegalActions(index):
##                    s = state.generateSuccessor(index,a)
##                    
##            
##            for a in state.getLegalActions(index):
##                s = state.generateSuccessor(index,a)
##                all_poss(index+1,s)
##            return
##                    
                    
        
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def fmax(depth,state,alpha,beta): #Returns (score,state), pacman only
            D= (-1)*float('inf')
            act = None
            if ((state.isWin())|(state.isLose())): #TERMINAL CONDITION MANZ
                return self.evaluationFunction(state)
            if (depth <=0):
                return state.getScore()
            for a in state.getLegalActions(0):
                s = state.generateSuccessor(0,a) # s is the new state
                score = fmin(s,1,depth,alpha,beta) # caries score of what all agents would do given my action
                if (score>D):
                    D = score
                    act = a
                if (D>beta):
                    return D
                alpha=max(alpha,D)
            if (depth==self.depth): #IF i'm in the upper call
                #print D
                return act
            #print D
            return D
                #generate all possible moves
                #from each move see pac's response
                                    
        def fmin(state,index,depth,alpha,beta):
            D = float('inf')
            if ((state.isWin())|(state.isLose())):
                return self.evaluationFunction(state)
            if ( index == (state.getNumAgents() -1)):
                #LAST AGENT
                for a in state.getLegalActions(index):
                    #HERE alpha is the maximum thing on MAx's options
                    s = state.generateSuccessor(index,a)
                    D = min(D,fmax(depth-1,s,alpha,beta)) # GO BACK TO PACMAN
                    if (D<alpha):
                        return D
                    beta= min(beta,D)

            else:
                for a in state.getLegalActions(index):
                    s = state.generateSuccessor(index,a)
                    D = min(D,fmin(s,index+1,depth,alpha,beta))
                    if (D<alpha):
                        return D
                    beta=min(beta,D)
            return D
        return fmax(self.depth,gameState,(-1)*float('inf'),float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def fmax(depth,state): #Returns (score,state), pacman only
            D= (-1)*float('inf')
            act = None
            if ((state.isWin())|(state.isLose())): #TERMINAL CONDITION MANZ
                return self.evaluationFunction(state)
            if (depth <=0):
                 return self.evaluationFunction(state)
            for a in state.getLegalActions(0):
                s = state.generateSuccessor(0,a) # s is the new state
                score = fmin(s,1,depth) # caries score of what all agents would do given my action
                if (score>D):
                    D = score
                    act = a
            if (depth==self.depth): #IF i'm in the upper call
                #print D
                return act
            #print D
            return D
                #generate all possible moves
                #from each move see pac's response
                                    
        def fmin(state,index,depth):
            D = float('inf')
            av = 0.0
            if ((state.isWin())|(state.isLose())):
                return self.evaluationFunction(state)
            G = state.getLegalActions(index)
            if ( index == (state.getNumAgents() -1)):
                #LAST AGENT
                for a in G:
                    s = state.generateSuccessor(index,a)
                    av += fmax(depth-1,s)# GO BACK TO PACMAN
            else:
                for a in G:
                    s = state.generateSuccessor(index,a)
                    av += fmin(s,index+1,depth)
            return float(av)/float(len(G))
        return fmax(self.depth,gameState)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #FOOD
    F,C,G,S,T = 0,0,0,0,0
    score = 0
    if (currentGameState.isWin()):
        return float('inf')
    elif (currentGameState.isLose()):
        return (-1)*float('inf')
    #print "HI"
    #print currentGameState.getPacmanState().getDirection()
    #if (currentGameState.getPacmanState().getDirection() == Directions.STOP):
     #   print "H"
      #  return (-1)*float('inf')
    mypos= currentGameState.getPacmanPosition()
    for i in currentGameState.getFood().asList():
        F += manhattanDistance(mypos,i)
    K = float('inf')
    for i in currentGameState.getCapsules():
        dist = manhattanDistance(mypos,i)
        K = min(dist,K)
        # We only need the min distance
        # the more the worse, just like food but a higher weighting
    for i in currentGameState.getGhostStates():
        if (i.scaredTimer>0):
            S += float(1.0)/float(manhattanDistance(mypos,i.getPosition())*i.scaredTimer)
        else:
            G+=manhattanDistance(mypos,i.getPosition()) #The more the better
    if (F>0):
        score += (float(1.0)/float(F))*20.0
    if (C>0):
        #print C
        #print (float(1.0)/float(C))
        score += (float(1.0)/float(K))*40
    #if (S>0):
     #   score += (float(1.0)/float(S))*1.0
    return score + G + S*10 + currentGameState.getScore()
    

# Abbreviation
better = betterEvaluationFunction

