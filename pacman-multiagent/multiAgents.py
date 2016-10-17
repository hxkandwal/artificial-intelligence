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


from util import manhattanDistance, Queue
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

        # choose the best evaluated value with only half probability, as sometimes a lesser optimal (non best) move
        # can yield good result.
        if random.random() > 0.5:
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        else:
            bestIndices = [index for index in range(len(scores)) if scores[index] > -5000]

        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # reflex agent should have better evaluation of the successor based on current parameters
        # its just heuristic so can be relaxed. using manhattan distance only to find nearest food.

        ghost_locations = [newGhostState.configuration.pos for newGhostState in newGhostStates]
        anticipated_ghost_locations = []

        # we have to check that the new position would be safer also from the next ghost states, this is
        # important because of random behaviour of ghost, we have to ensure that we are safe from all the
        # possible scenarios.
        for ghost_location in ghost_locations:
            (x, y) = ghost_location
            anticipated_ghost_locations.append(ghost_location)
            anticipated_ghost_locations.append((x + 1, y))
            anticipated_ghost_locations.append((x - 1, y))
            anticipated_ghost_locations.append((x, y + 1))
            anticipated_ghost_locations.append((x, y - 1))

        # if pacman is heading toward any ghost location, then its a bad move. (assigned a very high value)
        if newPos in anticipated_ghost_locations:
            return -10000

        # finding closest food location (manhattan distance, relaxed heuristic).
        min_distance = 10000

        for food_location in currentGameState.getFood().asList():
            manhattan_distance = manhattanDistance(successorGameState.getPacmanPosition(), food_location)

            if manhattan_distance < min_distance:
                min_distance = manhattan_distance

        # we have negate the min-distance from the current score value, as we have to travel at-least min-distance
        # steps to eat the nearest dot. And from the current metric that we are using, we know that we have a cost
        # -1 with every empty step taken.
        score = successorGameState.getScore() - min_distance
        return score


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

        # since, we are running or playing all the agents per one iteration (depth), so we have multiply depth with
        # the number of agents, to simulate depth-level look-ahead

        # return only the action.
        return self.value(gameState, gameState.getNumAgents() * self.depth, 0)[1]

    def value(self, gameState, depth, agent_index):
        """
            this method will more or less act like a dispatcher/recursion manager function, where I will explore and
            evaluate the min-max recursion call hierarchy accordingly (if any) (DFS way) or evaluate the result, if
            in case no further call is required.
        """

        # a very critical master check to determine, when not to recurse further (even when depth is remaining)
        if depth == 0 or \
                (agent_index <= gameState.getNumAgents() - 1 and len(gameState.getLegalActions(agent_index)) == 0) or \
                (agent_index > gameState.getNumAgents() - 1 and len(gameState.getLegalActions(0)) == 0):

            return (self.evaluationFunction(gameState), "NA")

        # invoke maximizer for agent_index = 0, else minimizer
        if agent_index == 0:
            return self.maximizer(gameState, depth, agent_index)
        else:
            # all the agents (ghosts) have played out, call pac-man again, as number of ghosts = (#agents - 1)
            if agent_index > gameState.getNumAgents() - 1:
                return self.maximizer(gameState, depth, 0)
            else:
                return self.minimizer(gameState, depth, agent_index)

    def maximizer(self, gameState, depth, agent_index):
        """
            a maximizer function
        """

        best_value = -10000000
        best_action = "NA"

        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)

            # recurse on the next level (min tree)
            action_value = self.value(successor_state, depth - 1, agent_index + 1)[0]

            if action_value >= best_value:
                best_value = action_value
                best_action = action

        return (best_value, best_action)

    def minimizer(self, gameState, depth, agent_index):
        """
            a minimizer function
        """

        best_value = 10000000
        best_action = "NA"

        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)

            # recurse on the next level (min tree, if agents are remaining else max tree)
            action_value = self.value(successor_state, depth - 1, agent_index + 1)[0]

            if action_value <= best_value:
                best_value = action_value
                best_action = action

        return (best_value, best_action)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax (with alpha-beta pruning) evaluated action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # since, we are running or playing all the agents per one iteration (depth), so we have multiply depth with
        # the number of agents, to simulate depth-level look-ahead

        # return only the action.
        return self.value(gameState, gameState.getNumAgents() * self.depth, 0, -float("inf"), float("inf"))[1]

    def value(self, gameState, depth, agent_index, alpha_value, beta_value):
        """
            this method will more or less act like a dispatcher/recursion manager function, where I will explore and
            evaluate the min-max (with alpha-beta pruning) recursion call hierarchy accordingly (if any) (DFS way) or
            evaluate the result, if in case no further call is required.
        """

        # a very critical master check to determine, when not to recurse further (even when depth is remaining)
        if depth == 0 or \
                (agent_index <= gameState.getNumAgents() - 1 and len(gameState.getLegalActions(agent_index)) == 0) or \
                (agent_index > gameState.getNumAgents() - 1 and len(gameState.getLegalActions(0)) == 0):

            return (self.evaluationFunction(gameState), "NA")

        # invoke maximizer for agent_index = 0, else minimizer
        if agent_index == 0:
            return self.maximizer(gameState, depth, agent_index, alpha_value, beta_value)
        else:
            # all the agents (ghosts) have played out, call pac-man again, as number of ghosts = (#agents - 1)
            if agent_index > gameState.getNumAgents() - 1:
                return self.maximizer(gameState, depth, 0, alpha_value, beta_value)
            else:
                return self.minimizer(gameState, depth, agent_index, alpha_value, beta_value)

    def maximizer(self, gameState, depth, agent_index, alpha_value, beta_value):
        """
            a maximizer function
        """

        best_value = -float("inf")
        best_action = "NA"

        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)

            # recurse on the next level (min tree)
            action_value = self.value(successor_state, depth - 1, agent_index + 1, alpha_value, beta_value)[0]

            if action_value >= best_value:
                best_value = action_value
                best_action = action

            # time to cut off, stop recursion if we have found something bigger than beta value
            if action_value > beta_value:
                return (best_value, best_action)

            if action_value > alpha_value:
                alpha_value = action_value

        return (best_value, best_action)

    def minimizer(self, gameState, depth, agent_index, alpha_value, beta_value):
        """
            a minimizer function
        """

        best_value = float("inf")
        best_action = "NA"

        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)

            # recurse on the next level (min tree, if agents are remaining else max tree)
            action_value = self.value(successor_state, depth - 1, agent_index + 1, alpha_value, beta_value)[0]

            if action_value <= best_value:
                best_value = action_value
                best_action = action

            # time to cut off, stop recursion if we have found something lesser than alpha value
            if action_value < alpha_value:
                return (best_value, best_action)

            if action_value < beta_value:
                beta_value = action_value

        return (best_value, best_action)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax searched action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # since, we are running or playing all the agents per one iteration (depth), so we have multiply depth with
        # the number of agents, to simulate depth-level look-ahead

        # return only the action.
        return self.value(gameState, gameState.getNumAgents() * self.depth, 0)[1]

    def value(self, gameState, depth, agent_index):
        """
            this method will more or less act like a dispatcher/recursion manager function, where I will explore and
            evaluate the expectiMax tree recursion call hierarchy accordingly (if any) (DFS way) or
            evaluate the result, if in case no further call is required.
        """

        # a very critical master check to determine, when not to recurse further (even when depth is remaining)
        if depth == 0 or \
                (agent_index <= gameState.getNumAgents() - 1 and len(gameState.getLegalActions(agent_index)) == 0) or \
                (agent_index > gameState.getNumAgents() - 1 and len(gameState.getLegalActions(0)) == 0):

            return (self.evaluationFunction(gameState), "NA")

        # invoke maximizer for agent_index = 0, else minimizer
        if agent_index == 0:
            return self.maximizer(gameState, depth, agent_index)
        else:
            # all the agents (ghosts) have played out, call pac-man again, as number of ghosts = (#agents - 1)
            if agent_index > gameState.getNumAgents() - 1:
                return self.maximizer(gameState, depth, 0)
            else:
                return self.minimizer(gameState, depth, agent_index)

    def maximizer(self, gameState, depth, agent_index):
        """
            a maximizer function
        """

        best_value = -float("inf")
        best_action = "NA"

        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)

            # recurse on the next level (min tree)
            action_value = self.value(successor_state, depth - 1, agent_index + 1)[0]

            if action_value >= best_value:
                best_value = action_value
                best_action = action

        return (best_value, best_action)

    def minimizer(self, gameState, depth, agent_index):
        """
            a expecti-minimizer function which does the average of all the below held action outcomes
            i.e. (0.5 probability)
        """

        best_value = 0
        best_action = "NA"

        action_value_summation = 0

        for action in gameState.getLegalActions(agent_index):
            successor_state = gameState.generateSuccessor(agent_index, action)

            # recurse on the next level (min tree, if agents are remaining else max tree)
            action_value = self.value(successor_state, depth - 1, agent_index + 1)[0]

            action_value_summation += action_value

        best_value = action_value_summation/len(gameState.getLegalActions(agent_index))
        return (best_value, best_action)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      As described in the class, I created a linear evaluation function, which can be explained as weighted linear sum
      of the participating features. For Example :

            Eval(s) = w1 * f1(s) + w2 * f2(s) + w3 * f3(s) + w4 * f4(s) + ........ + wn * fn(s)

      where, s here represents the state.

      In my evaluation function design, I am keeping currentGameState.getScore() as the dominant feature however, there
      are other features which can be modelled as :

      Fp = positive features, like food, capsules and scared ghosts
           Function computes the sum of the inverses of the Manhattan distances to all the food and edible ghosts spots.

           1/ manhattan-distance (current-pos, food1) + 1/ manhattan-distance (current-pos, food2) .... +
           1/ manhattan-distance (current-pos, scared_ghost1) + ..... + 1/ manhattan-distance (current-pos, scared_ghostN)

      Fn = negative features, like ghosts.
           Function computes the sum of the inverses of the Manhattan distances to all the ghosts spots.

           1/ manhattan-distance (current-pos, ghost1) + ..... + 1/ manhattan-distance (current-pos, ghostN)

      With this approach, more closer the food or a scared ghost is, more likely we are going to advance for it since
      its feature Fp is inversely proportional. Also, using the negation with the Fn, more closer the ghost is, less
      likely we will advance towards it.
      
      So, our evaluation function would be like :

            Eval(s) = currentGameState.getScore() + Wp * Fp(s) - Wp * Fn(s)

      where, Wp and Wn are the associated empirical weights, chosen as per to improve performance.

      Wp = 1 Wn = 1    Pac-man wins without much performance.
      Wp < 5 Wn = 1    Pac-man fails in some simulations. This meant I had to increase negative weights proportionally.
      Wp = 4 Wn = 3    Pac-man performs to the grader and test cases standards.

      Also, edible scared ghosts are given more weights (5) as its more favourable to eat those as compared to food
      pellets, as that will clear way and remove the future threats.

      This is because firstly, scaredness of scared ghosts is time-bound and second, after eating they will spawn from a
      (probably) distant location.

    """
    "*** YOUR CODE HERE ***"

    # positive feature
    aggregate_inverse_food_distance = 0

    # added 1 to avoid ZeroDivisionError.
    for food_location in currentGameState.getFood().asList():
        aggregate_inverse_food_distance += 1.0/(1 + manhattanDistance(currentGameState.getPacmanPosition(), food_location))

    for ghost_state in currentGameState.getGhostStates():
        if ghost_state.scaredTimer > 0:
            manhattan_distance = manhattanDistance(currentGameState.getPacmanPosition(), ghost_state.getPosition())

            if ghost_state.scaredTimer >= manhattan_distance:
                aggregate_inverse_food_distance += 5.0/(1 + manhattan_distance)

    # negative feature
    aggregate_inverse_ghosts_distance = 0

    # added 1 to avoid ZeroDivisionError.
    for ghost_location in [newGhostState.configuration.pos for newGhostState in currentGameState.getGhostStates()]:
        aggregate_inverse_ghosts_distance += 1.0/(1 + manhattanDistance(currentGameState.getPacmanPosition(), ghost_location))

    # calculating the evaluated score
    score = currentGameState.getScore() + 4 * aggregate_inverse_food_distance - 3 * aggregate_inverse_ghosts_distance

    return score


# Abbreviation
better = betterEvaluationFunction
