# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    visited_states = set()

    actions_stack = Stack()
    depthFirstSearchInner(problem, visited_states, actions_stack, problem.getStartState())

    # reverse the action stack
    actions_stack = actions_stack.list[::-1]
    return actions_stack


def depthFirstSearchInner(problem, visited_states, actions_stack, current_state):
    visited_states.add(current_state)

    if problem.isGoalState(current_state):
        print "goal state :", current_state
        return True

    successors = problem.getSuccessors(current_state)

    if len(successors) > 0:
        for successor in reversed(successors):
            if successor[0] not in visited_states:
                if depthFirstSearchInner(problem, visited_states, actions_stack, successor[0]):
                    actions_stack.push(successor[1])
                    return True
    return False


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # this will contain the visited states (closed/visited nodes)
    visited_states = set()

    # we have to use PriorityQueue, which is a min-heap and keep minimum value node on the top.
    fringe_queue = Queue()
    actions = []

    # fringe tuple structure : current state, actions_stack, path effective cost
    fringe_queue.push((problem.getStartState(), actions))

    # execute BFS
    while not fringe_queue.isEmpty():
        fringe_tuple = fringe_queue.pop()
        current_state, current_actions_stack = fringe_tuple

        if problem.isGoalState(current_state):
            return current_actions_stack

        # check if current_state is not already visited before (graph search)
        if current_state not in visited_states:
            visited_states.add(current_state)

            # as current state is not the goal state and not explored yet
            # so expand the current node to queue the successors in the fringe.
            successors = problem.getSuccessors(current_state)

            for successor in successors:
                successor_state, successor_direction, successor_cost = successor

                successor_action_stack = current_actions_stack[:]
                successor_action_stack.append(successor_direction)
                fringe_queue.push((successor_state, successor_action_stack))

    return actions


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # this will contain the visited states (closed/visited nodes)
    visited_states = set()

    # we have to use PriorityQueue, which is a min-heap and keep minimum value node on the top.
    fringe_queue = PriorityQueue()
    actions = []

    # fringe tuple structure : current state, actions_stack, path effective cost
    fringe_queue.push((problem.getStartState(), actions, 0), 0)

    # execute UCS
    while not fringe_queue.isEmpty():
        fringe_tuple = fringe_queue.pop()
        current_state, current_actions_stack, current_path_cost = fringe_tuple

        if problem.isGoalState(current_state):
            return current_actions_stack

        # check if current_state is not already visited before (graph search)
        if current_state not in visited_states:
            visited_states.add(current_state)

            # as current state is not the goal state and not explored yet
            # so expand the current node to queue the successors in the fringe.
            successors = problem.getSuccessors(current_state)

            for successor in successors:
                successor_state, successor_direction, successor_cost = successor

                successor_action_stack = current_actions_stack[:]
                successor_action_stack.append(successor_direction)
                successor_effective_cost = current_path_cost + successor_cost
                fringe_queue.push((successor_state, successor_action_stack, successor_effective_cost), successor_effective_cost)

    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # this will contain the visited states (closed/visited nodes)
    visited_states = set()

    # we have to use PriorityQueue, which is a min-heap and keep minimum value node on the top.
    fringe_queue = PriorityQueue()
    actions = []

    # fringe tuple structure : current state, actions_stack, path effective cost
    fringe_queue.push((problem.getStartState(), actions, 0), 0 + heuristic(problem.getStartState(), problem))

    # execute A*
    while not fringe_queue.isEmpty():
        fringe_tuple = fringe_queue.pop()
        current_state, current_actions_stack, current_path_cost = fringe_tuple

        if problem.isGoalState(current_state):
            return current_actions_stack

        # check if current_state is not already visited before (graph search)
        if current_state not in visited_states:
            visited_states.add(current_state)

            # as current state is not the goal state and not explored yet
            # so expand the current node to queue the successors in the fringe.
            successors = problem.getSuccessors(current_state)

            for successor in successors:
                successor_state, successor_direction, successor_cost = successor

                successor_action_stack = current_actions_stack[:]
                successor_action_stack.append(successor_direction)
                # for A*, we also consider the heuristic cost.
                # f(n) = g(n) + h(n)
                successor_effective_cost = current_path_cost + successor_cost + heuristic(successor_state, problem)
                fringe_queue.push((successor_state, successor_action_stack, current_path_cost + successor_cost), successor_effective_cost)

    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
