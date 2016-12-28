from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.Q = {}

  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    return self.Q.get((state, action), 0.0)

  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    pacman_state = state.getPacmanState()
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return 0.0

    # Iterate over all the legal actions we have and find the max Q from it
    best_q = -sys.maxsize
    for action in legalActions:
      new_pos = Actions.getSuccessor(pacman_state.getPosition(), action)
      current_q = self.Q.get((pacman_state.getPosition(), new_pos), 0.0)
      if current_q > best_q:
        best_q = current_q

    if best_q == -sys.maxsize:
      best_q = 0.0

    return best_q

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    pacman_state = state.getPacmanState()
    legalActions = self.getLegalActions(state)
    best_q = -sys.maxsize
    best_action = None
    for action in legalActions:
      new_pos = Actions.getSuccessor(pacman_state.getPosition(), action)
      current_q = self.Q.get((pacman_state.getPosition(), new_pos), 0.0)
      if current_q > best_q:
        best_q = current_q
        best_action = action

    action = best_action

    return action

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    # Pick Action
    probability = random.random()
    legalActions = self.getLegalActions(state)

    if probability > self.epsilon:
      action = random.choice(legalActions)
    else:
      # Find the best action from the Q values computed so far
      action = self.getPolicy(state)

    return action

  def getReward(self, state):
    pacman_state = state.getPacmanState()
    pos = pacman_state.getPosition()

    if state.isLose():
      return -1000.0

    if state.isWin():
      return 1000.0

    reward = 0.0
    if state.hasFood(pos[0], pos[1]):
      reward += 1.0

    capsules = state.getCapsules()
    if len(capsules) > 0 and capsules[pos[0]][pos[1]] is True:
      reward += 10.0

    ghost_states = state.getGhostStates()

    # We have already checked for win or lose condition so the only
    # way a ghost and player can share a position is if the ghost is
    # scared
    for ghost_state in ghost_states:
      if ghost_state.getPosition() == pos:
        reward += 50.0

    return reward

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    pacman_state = state.getPacmanState()
    current_pos = pacman_state.getPosition()
    new_pos = Actions.getSuccessor(current_pos, action)
    current_val = self.Q.get((current_pos, new_pos), 0.0)

    # Transition to the new pos and find the best policy for it
    pacman_state.pos = new_pos
    self.Q[(current_pos, new_pos)] = self.alpha * (self.getReward(state) + self.getValue(state) - current_val)


class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action

    
class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
