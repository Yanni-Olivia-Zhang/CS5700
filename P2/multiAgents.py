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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** BEGIN YOUR CODE HERE ***"
        # don't run into ghost
        newGhostPositions = successorGameState.getGhostPositions()
        for posi in newGhostPositions:
            distToGhost = manhattanDistance(newPos, posi)
            if distToGhost <= 2:
                return 0
            
        # try to eat food pellets
        foodList = currentGameState.getFood().asList()
        newFoodList = newFood.asList()
        if len(newFoodList) < len(foodList):
            return 99999
        minDis = manhattanDistance(newPos, foodList[0])
        for food in newFoodList:
            distToFood = manhattanDistance(newPos, food)
            if distToFood < minDis:
                minDis = distToFood
        return 1/minDis
        "*** END YOUR CODE HERE ***"
        # ^^^ you should return something in the above block
        
        # but by default, this is the evaluation function before you put your code in
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** BEGIN YOUR CODE HERE ***"
        #util.raiseNotDefined()
        self.minimax(gameState)
        return self.action
    
    def minimax(self, gameState, agentIndex = 0, depth = 0):
        agentNum = gameState.getNumAgents() - 1
        if agentIndex > agentNum:
            agentIndex = 0
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth + 1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agentIndex, depth)
                       
    def maxValue(self, gameState, agentIndex, depth):
        v = -999999
        dic = {}
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            dic[action] = self.minimax(successorState, agentIndex + 1, depth)
            v = max(v, dic[action])
        if depth == 1:
            for action in dic:
                if dic[action] == v:
                    self.action = action
        return v
    
    def minValue(self, gameState, agentIndex, depth):
        v = 999999
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.minimax(successorState, agentIndex + 1, depth))
        return v
        "*** END YOUR CODE HERE ***"

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** BEGIN YOUR CODE HERE ***"
        #util.raiseNotDefined()
        self.minimax(gameState)
        return self.action
    
    def minimax(self, gameState, agentIndex = 0, depth = 0, alpha = -999999, beta = 999999):
        agentNum = gameState.getNumAgents() - 1
        if agentIndex > agentNum:
            agentIndex = 0
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth + 1, alpha, beta)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)
                       
    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        v = -999999
        dic = {}
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            dic[action] = self.minimax(successorState, agentIndex + 1, depth, alpha, beta)
            v = max(v, dic[action])
            if v > beta:
                return v
            alpha = max(alpha, v)
            if depth == 1:
                for action in dic:
                    if dic[action] == v:
                        self.action = action
        return v
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        v = 999999
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.minimax(successorState, agentIndex + 1, depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)  
        return v
        "*** END YOUR CODE HERE ***"

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** BEGIN YOUR CODE HERE ***"
        #util.raiseNotDefined()
        self.expectimax(gameState)
        return self.action
    def expectimax(self, gameState, agentIndex = 0, depth = 0):
        agentNum = gameState.getNumAgents() - 1
        if agentIndex > agentNum:
            agentIndex = 0
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth + 1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.expValue(gameState, agentIndex, depth)
        
    def maxValue(self, gameState, agentIndex, depth):
        v = -999999
        dic = {}
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            dic[action] = self.expectimax(successorState, agentIndex + 1, depth)
            v = max(v, dic[action])
        if depth == 1:
            for action in dic:
                if dic[action] == v:
                    self.action = action
        return v
    
    def expValue(self, gameState, agentIndex, depth):
        v = 0
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            p = 1/len(gameState.getLegalActions(agentIndex))
            v += p * self.expectimax(successorState, agentIndex + 1, depth)
        return v
        "*** END YOUR CODE HERE ***"

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    pac = currentGameState.getPacmanPosition()
    foodcount = currentGameState.getNumFood()
    currentScore = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    numCapsules = len(currentGameState.getCapsules())

    foodList = currentGameState.getFood().asList()
    minD = min(manhattanDistance(pac, food_pos) for food_pos in foodList) if foodList else 0

    ghostEval = 0
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        dis = manhattanDistance(pac, ghostPos)
        if ghost.scaredTimer > dis:
            ghostEval += 100 - dis
        else:
            ghostEval -= 2 ** (8 - dis)

    if numCapsules > 0:
        capsuleWeight = 2 * numCapsules
    else:
        capsuleWeight = 0

    remainingFoodWeight = -10 * foodcount
    totalScoreWeight = currentScore
    totalCapsulesWeight = capsuleWeight
    closestFoodWeight = -2 * minD
    ghostWeight = ghostEval

    return totalScoreWeight + remainingFoodWeight + totalCapsulesWeight + closestFoodWeight + ghostWeight

    "*** END YOUR CODE HERE ***"

# Abbreviation
better = betterEvaluationFunction
