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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

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
        newPacmanPos = successorGameState.getPacmanPosition()
        oldFoodPos = currentGameState.getFood().asList()
        newFoodPos = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()

        # CODE STARTS HERE

        # Gather food statistics
        foodLeft = len(newFoodPos)
        oldFoodLeft = len(oldFoodPos)
        foodDists = [ manhattanDistance(foodPos, newPacmanPos) for foodPos in newFoodPos ]

        if foodLeft > 0:
            minFoodDist = min(foodDists)
        else:
            minFoodDist = 0

        dotEaten = (oldFoodLeft - foodLeft)

        # Gather ghost statistics
        newGhostPos = [nextGhostState.getPosition() for nextGhostState in newGhostStates]
        ghostDists = [ manhattanDistance(ghostPos, newPacmanPos) for ghostPos in newGhostPos]

        # Do magic calculations
        res = 0
        if min(ghostDists) <= 2:
            res -= sum(map(self.inverse, ghostDists))

        if dotEaten:
            res += 100

        res -= (minFoodDist)

        return res

    def inverse(self, x):
        if x == 0:
            return 1000
        else:
            return 200/x

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """

        return self.maximize(gameState, 1, 0)[1]


    def getNodeValue(self, gameState, currDepth, agentIndex):
        # Check if game is a winning or loosing state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)


        # Check if more agents are available in the current layer
        if agentIndex == gameState.getNumAgents():
            # Reset agent to 0 (Pacman) & increase searched depth
            agentIndex = 0
            currDepth += 1

            # Check if no more recursion is needed (self.depth is reached)
            if currDepth > self.depth:
                # Return the result of the evalution function
                return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maximize(gameState, currDepth, agentIndex)[0]
        else:
            return self.minimize(gameState, currDepth, agentIndex)[0]

    def getSuccessorValues(self, gameState, currDepth, agentIndex):
        """ Returns a list of tuples (successorValue, actionUsed) """

        legalActions = gameState.getLegalActions(agentIndex)
        successorNodes = [ (gameState.generateSuccessor(agentIndex, action), action)
                                for action in legalActions ]

        return [ (self.getNodeValue(successor, currDepth, agentIndex + 1), action)
                                for (successor, action) in successorNodes ]

    def maximize(self, gameState, currDepth, agentIndex):
        """ Maximizing agent """
        successorValues = self.getSuccessorValues(gameState, currDepth, agentIndex)
        return max(successorValues)


    def minimize(self, gameState, currDepth, agentIndex):
        """ Minimizing agent """
        successorValues = self.getSuccessorValues(gameState, currDepth, agentIndex)
        return min(successorValues)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    NO_ACTION = 'NO_ACTION'
    NODE_MIN = -(1e9)
    NODE_MAX =  (1e9)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maximize(gameState, 1, 0, self.NODE_MIN, self.NODE_MAX)[1]

    def getNodeValue(self, gameState, currDepth, agentIndex, alpha, beta):
        # Check if game is a winning or loosing state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Check if more agents are available in the current layer
        if agentIndex == gameState.getNumAgents():
            # Reset agent to 0 (Pacman) & increase searched depth
            agentIndex = 0
            currDepth += 1

            # Check if no more recursion is needed (self.depth is reached)
            if currDepth > self.depth:
                # Return the result of the evalution function
                return self.evaluationFunction(gameState)

        if agentIndex == 0:
            # Agent is Pacman!
            return self.maximize(gameState, currDepth, agentIndex, alpha, beta)[0]
        else:
            # Agent is Ghost
            return self.minimize(gameState, currDepth, agentIndex, alpha, beta)[0]

    def maximize(self, gameState, currDepth, agentIndex, alpha, beta):
        """ Maximizing agent. Returns tuple (maxNodeValue, actionUsed) """

        legalActions = gameState.getLegalActions(agentIndex)

        # Use first successor to initialize the current nodeValue
        nodeValue = self.NODE_MIN
        bestAction = self.NO_ACTION

        # Recursively loop through the remaining successors
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.getNodeValue(successor, currDepth, agentIndex + 1, alpha, beta)
            if successorValue > nodeValue:
                nodeValue = successorValue
                bestAction = action

            # Check if we can prune. NOTE: MAXimizing agent -> check beta!
            if nodeValue > beta:
                break

            # MAXimizing agent -> update alpha!
            alpha = max(alpha, nodeValue)

        return (nodeValue, bestAction)

    def minimize(self, gameState, currDepth, agentIndex, alpha, beta):
        """ Minimizing agent. Returns tuple (minNodeValue, actionUsed) """

        legalActions = gameState.getLegalActions(agentIndex)

        nodeValue = self.NODE_MAX
        bestAction = self.NO_ACTION

        # Recursively loop through the remaining successors
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.getNodeValue(successor, currDepth, agentIndex + 1, alpha, beta)
            if successorValue < nodeValue:
                nodeValue = successorValue
                bestAction = action

            # Check if we can prune. NOTE: MINimizing agent -> check alpha!
            if nodeValue < alpha:
                break

            # MINimizing agent -> update beta!
            beta = min(beta, nodeValue)

        return (nodeValue, bestAction)

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
        return self.maximize(gameState, 1, 0)[1]

    def getNodeValue(self, gameState, currDepth, agentIndex):
        # Check if game is a winning or loosing state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)


        # Check if more agents are available in the current layer
        if agentIndex == gameState.getNumAgents():
            # Reset agent to 0 (Pacman) & increase searched depth
            agentIndex = 0
            currDepth += 1

            # Check if no more recursion is needed (self.depth is reached)
            if currDepth > self.depth:
                # Return the result of the evalution function
                return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maximize(gameState, currDepth, agentIndex)[0]
        else:
            return self.average(gameState, currDepth, agentIndex)

    def getSuccessorValues(self, gameState, currDepth, agentIndex):
        """ Returns a list of tuples (successorValue, actionUsed) """

        legalActions = gameState.getLegalActions(agentIndex)
        successorNodes = [ (gameState.generateSuccessor(agentIndex, action), action)
                                for action in legalActions ]

        return [ (self.getNodeValue(successor, currDepth, agentIndex + 1), action)
                                for (successor, action) in successorNodes ]

    def maximize(self, gameState, currDepth, agentIndex):
        """ Maximizing agent. Returns (maxSuccessorValue, actionUsed) """
        successorValues = self.getSuccessorValues(gameState, currDepth, agentIndex)
        maxValue = max(successorValues)[0]

        equalBest = [ action for (value, action) in successorValues if value == maxValue ]
        randSelection = random.randint(0, len(equalBest) - 1)
        return (maxValue, equalBest[randSelection])


    def average(self, gameState, currDepth, agentIndex):
        """ Expected agent. Returns float (expectedValue) """
        successorValues = [ value[0] for value in self.getSuccessorValues(gameState, currDepth, agentIndex)]
        return sum(successorValues) * 1.0 / len(successorValues)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    ghostStates = currentGameState.getGhostStates()
    pacmanPos = currentGameState.getPacmanPosition()

    if currentGameState.isWin():
        return 10e9
    elif currentGameState.isLose():
        return -10e9

    # Calculate food stats
    # *************************************************************
    allFoodGrid = currentGameState.getFood()
    foodCount = currentGameState.getNumFood()

    # FoodPellets left
    gridSize = allFoodGrid.width * allFoodGrid.height
    foodLeftScore = 100.0 * (gridSize - foodCount) / gridSize

    # Closest pellet
    foodList = allFoodGrid.asList()
    closestFoodPellet = min( manhattanDistance(pacmanPos, foodPos) for foodPos in foodList )
    #maxGridDist = allFoodGrid.width + allFoodGrid.height

    #walls = currentGameState.getWalls()
    #closestFoodPelletDist = closestFoodPellet[0] + calcWallPenalty(walls, pacmanPos, closestFoodPellet[1])
    # **************************************************************

    #ghostPos = [ghostState.getPosition() for ghostState in ghostStates]
    #ghostDists = [ manhattanDistance(ghostPos, pacmanPos) for ghostPos in ghostPos]

    totalScore = -closestFoodPellet + 10e5*currentGameState.getScore()
    #print '%s: %5f %5f %5f-> %f ' % (pacmanPos, foodLeftScore, foodClosestScore, currentScore, totalScore)

    return totalScore

def calcWallPenalty(walls, pacmanPos, pelletPos):
    penalty = 0
    for x in range(min(pacmanPos[0], pelletPos[0]), max(pacmanPos[0], pelletPos[0])):
        if (x, pacmanPos[1]) in walls or (x, pelletPos[1]) in walls:
            penalty += 1

    for y in range(min(pacmanPos[1], pelletPos[1]), max(pacmanPos[1], pelletPos[1])):
        if (pacmanPos[0], y) in walls or (pelletPos[0], y) in walls:
            penalty += 1

    return penalty



# Abbreviation
better = betterEvaluationFunction

