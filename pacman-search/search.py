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
    print "actions path :", actions_stack
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
    # this will contain child-parent successor pair
    visited_state_tuples = set()

    fringe_queue = Queue()
    fringe_queue.push(((problem.getStartState(), None, None), None))

    # execute BFS
    goal_tuple = breadthFirstSearchInner(problem, visited_state_tuples, fringe_queue)
    print "goal_tuple :", goal_tuple

    actions_stack = Stack()

    # back tracking to build action stack, will be build bottom-up.
    back_tracked_tuple = goal_tuple

    while back_tracked_tuple[1] is not None:
        actions_stack.push(back_tracked_tuple[1])
        for visited_state_tuple in visited_state_tuples:
            if visited_state_tuple[0] == back_tracked_tuple:
                back_tracked_tuple = visited_state_tuple[1]
                break

    # reverse the action stack
    actions_stack = actions_stack.list[::-1]
    print "actions path :", actions_stack
    return actions_stack


def breadthFirstSearchInner(problem, visited_state_tuples, fringe_queue):
    future_fringe_queue = Queue()

    while len(fringe_queue.list) > 0:
        states_tuple = fringe_queue.pop()
        current_state = states_tuple[0][0]

        if current_state not in [visited_state_tuple[0][0] for visited_state_tuple in visited_state_tuples]:
            visited_state_tuples.add(states_tuple)

            if problem.isGoalState(current_state):
                print "goal state :", current_state
                return states_tuple[0]

            successors = problem.getSuccessors(current_state)

            if len(successors) > 0:
                for successor in successors:
                    if successor[0] not in \
                            [visited_state_tuple[0][0] for visited_state_tuple in visited_state_tuples]:
                        # add parent-child pair that will help to backtrack
                        future_fringe_queue.push((successor, states_tuple[0]))

    if len(future_fringe_queue.list) > 0:
        return breadthFirstSearchInner(problem, visited_state_tuples, future_fringe_queue)
    else:
        return None


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # this will contain child-parent successor pair
    visited_state_tuples = set()

    # we have to use PriorityQueue, which is a min-heap and keep minimum value node on the top.
    fringe_queue = PriorityQueue()

    # element structure : current node, parent node, effective cost
    fringe_queue.push(((problem.getStartState(), None, 0), None, 0), 0)

    # execute UFS
    goal_tuple = uniformCostSearchInner(problem, visited_state_tuples, fringe_queue)
    print "goal_tuple :", goal_tuple

    actions_stack = Stack()

    # back tracking to build action stack, will be build bottom-up.
    back_tracked_tuple = goal_tuple

    while back_tracked_tuple[1] is not None:
        actions_stack.push(back_tracked_tuple[1])
        for visited_state_tuple in visited_state_tuples:
            if visited_state_tuple[0] == back_tracked_tuple:
                back_tracked_tuple = visited_state_tuple[1]
                break

    # reverse the action stack
    actions_stack = actions_stack.list[::-1]
    print "actions path :", actions_stack
    return actions_stack


def uniformCostSearchInner(problem, visited_state_tuples, fringe_queue):

    while len(fringe_queue.heap) > 0:
        states_tuple = fringe_queue.pop()
        current_state = states_tuple[0][0]

        if current_state not in [visited_state_tuple[0][0] for visited_state_tuple in visited_state_tuples]:
            visited_state_tuples.add(states_tuple)

            if problem.isGoalState(current_state):
                print "goal state :", current_state
                return states_tuple[0]

            successors = problem.getSuccessors(current_state)

            if len(successors) > 0:
                for successor in successors:
                    if successor[0] not in \
                            [visited_state_tuple[0][0] for visited_state_tuple in visited_state_tuples]:
                        # add parent-child pair that will help to backtrack
                        fringe_queue.push((successor, states_tuple[0], states_tuple[0][2] + successor[2]), states_tuple[2] + successor[2])

    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
