import tensorflow as tf
import numpy as np
import random as rnd

from board import Board, moves_string, move_string
from engines.random import RandomEngine

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# We build a standard Multi-Layer Percepton, where the input is the
# state of the board, and the output is the expected reward for taking
# a particular action


def sigmoid(x):
    """
    Calculates the sigmoid of the given input
    :param x: the input tensor
    :return: the sigmoid of the tensor
    """
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

def sigmoid_derivative(x):
    """
    Calculates the derivative of the sigmoid, given the input
    :param x: the input tensor
    :return: the derivative of the sigmoid
    """
    return tf.multiply(sigmoid(x), tf.subtract(tf.constant(1.0), sigmoid(x)))

def greedy_policy(output):
	"""
	Returns a q value and the index of the action associated with it,
	according to the greedy policy
	
	:param ouput: the output vector of the forward pass
	:return: the index of the maximal q value, i.e. the index of the action
	and the maximal q value itself
	"""

	policy_action = output.argmax()
	policy_q = output[0, policy_action]
	
	return policy_action, policy_q
	
	
def epsilon_greedy_policy(epsilon, output, legal_actions):
	"""
	Returns a q value and the index of the action associated with it,
	according to the epsilon-greedy policy
	
	:param epsilon: the probability to choose random action
	:param output: the output vector of the forward pass
	:param legal_actions: the list of legal actions
	:return: the index of the policy-choosen q value, i.e. the index of the
	action and the q value associated with that
	"""
	
	policy_action = 0
	policy_q = 0

	if np.random.uniform(0,1) > 1 - epsilon:
		# choose a random action
		legal_actions_idxs = np.argwhere(legal_actions == 1)
		random_action_idx = np.random.randint(0, legal_actions_idxs.shape[0])
	
		policy_action = legal_actions_idxs[random_action_idx, 1];
		policy_q = output[0, policy_action]
	
	else:
		#choose the action giving maximal q value
		policy_action, policy_q = greedy_policy(output)
		
	return policy_action, policy_q 


def read_state(board):
	"""
	Reads every square of the board object and inserts it
	in the list representing the state. Squares in the board
	are processed column by column.
	
	:param board: instance of the class Board
	:return: state of the board, represented as a list
	"""
	state = [float(item) for sublist in board[:] for item in sublist]
	return np.array(state)[np.newaxis]
	
def board_to_nn_actions(board_actions):
	"""
	Transforms the actions from board representation to a representation
	suitable for the neural network.
	
	:param board_actions: list of tuples as actions in a board format
	:return: list of actions in a neural network format
	"""
	
	nn_actions = []
	for (x, y) in board_actions:
		nn_actions.append(x*8 + y)

	return nn_actions
	
def nn_to_board_action(nn_action):
	"""
	Transforms the action from neural network representation to a representation
	suitable for the board
	
	:param nn_action: single action in a neural network format
	:return: tuple as an action in a board format
	"""
	return (nn_action/8, nn_action % 8)


def winner(board):
    """ 
	Determine the winner of a given board.
	
	:param board: The board object
	:return: -1 for black player win, 1 for white player win
			 and 0 for draw
	"""
    black_count = board.count(-1)
    white_count = board.count(1)
	
    if black_count > white_count:	
        return -1
    elif white_count > black_count:			
        return 1
    else:
        return 0

########################################################################################################################
""" Forward Pass: Begin """

 # the number of input units, the board is 8 by 8
num_input_units = 8 * 8

# the number of units in the hidden layer
num_units_hidden_layer = 50;

# the number of output units, where every unit is the expected reward
# when taking the appropriate action, given the current state as input
num_output_units = 8 * 8

""" Initializing the Placeholders """

# placeholder for the input state, where -1 means a black disc,
# 1 means a white disc, and 0 means an empty field
X = tf.placeholder(tf.float32, [None, num_input_units])

# placeholder for the legal actions, where legal actions are denoted
# with 1 and illegal wih 0
legal_actions = tf.placeholder(tf.float32, [None, num_output_units])

""" Initializing the Variables """

# The first set of weights, i.e. from the input layer to the first hidden layer
W_1 = tf.Variable(tf.truncated_normal(shape=[num_input_units, num_units_hidden_layer], name="W_1"))

# The first set of biases
b_1 = tf.Variable(tf.truncated_normal(shape=[1, num_units_hidden_layer], name="b_1"))

# The second set of weights, i.e. from the hidden layer to the output layer
W_2 = tf.Variable(tf.truncated_normal(shape=[num_units_hidden_layer, num_output_units], name="W_2"))

# The second set of biases
b_2 = tf.Variable(tf.truncated_normal(shape=[1, num_output_units], name="b_2"))

""" Build the Computational Graph """

# Hidden Layer Values
z_1 = tf.add(tf.matmul(X, W_1), b_1, name="z_1")
# Hidden Layer Activations
a_1 = sigmoid(z_1)

# Output Layer Values
z_2 = tf.add(tf.matmul(a_1, W_2), b_2, name="z_2")

# Output Layer Activations
a_2_1 = sigmoid(z_2)

# Output Layer Link Function
a_2_2 = tf.multiply(legal_actions, a_2_1)

""" Forward Pass: End """
########################################################################################################################


########################################################################################################################
""" Backward Pass: Begin """

# Placeholder for the cost, since it is calculated as a difference
# between two consecutive forward passes
cost = tf.placeholder(tf.float32, [None, num_output_units])
z_2_eval = tf.placeholder(tf.float32, [None, num_output_units])
a_1_eval = tf.placeholder(tf.float32, [None, num_units_hidden_layer])
z_1_eval = tf.placeholder(tf.float32, [None, num_units_hidden_layer])

""" Calculate the weight gradients """

# Calculate gradients for the second set of weights
d_z_2 = tf.multiply(legal_actions, tf.multiply(cost, sigmoid_derivative(z_2_eval)))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1_eval), d_z_2)

# Calculate gradients for the first set of weights
d_a_1 = tf.matmul(d_z_2, tf.transpose(W_2))
d_z_1 = tf.multiply(d_a_1, sigmoid_derivative(z_1_eval))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(X), d_z_1)


""" Update the weights and biases """

eta = tf.constant(0.01)
step = [
    tf.assign(W_1, tf.subtract(W_1, tf.multiply(eta, d_w_1))),
    tf.assign(b_1, tf.subtract(b_1, tf.multiply(eta, tf.reduce_mean(d_b_1, axis=[0])))),
    tf.assign(W_2, tf.subtract(W_2, tf.multiply(eta, d_w_2))),
    tf.assign(b_2, tf.subtract(b_2, tf.multiply(eta, tf.reduce_mean(d_b_2, axis=[0]))))
]


""" Backward Pass: End """
########################################################################################################################


def play_game(nn_color):
	"""
	The neural network plays a game with the random engine,
	in order to evaulate if it is learning something.
	
	:param nn_color: the color of the Neural Network
	:return: 1 for win, 0 for loosing and 0.5 for draw
	"""
	
	score = 0
	b = Board()
	r_engine = RandomEngine()
	r_engine_color = nn_color * (-1)
	
	colors = [-1, 1]
	turn = 0
	for move_num in range(60):
		
		# The game is over
		if not b.get_legal_moves(nn_color) and not b.get_legal_moves(r_engine_color):				
			break
		else:
			# Neural Network turn
			if nn_color == colors[turn % 2]:
				
				# If not legal moves, jump
				if not b.get_legal_moves(nn_color):
					turn = turn + 1
					continue
				
				# Read the current state and the legal actions
				nn_state = read_state(b)
				nn_legal_actions_idxs = board_to_nn_actions(b.get_legal_moves(nn_color))
				nn_legal_actions = np.zeros(num_output_units)[np.newaxis]
				nn_legal_actions[0][nn_legal_actions_idxs] = 1
				
				# Evaluate the state
				nn_output = sess.run(a_2_2, feed_dict={X: nn_state, legal_actions: nn_legal_actions})
				
				# Choose the action with maximal q value and execute that action
				nn_max_action, _ = greedy_policy(nn_output)
				b.execute_move(nn_to_board_action(nn_max_action), nn_color)
				
			# Random Engine turn
			else:
				# If not legal moves, jump
				if not b.get_legal_moves(r_engine_color):
					turn = turn + 1
					continue
					
				r_action = r_engine.get_move(b, r_engine_color, move_num)
				b.execute_move(r_action, r_engine_color)
				
			turn = turn + 1
			
	# Return a score
	w = winner(b)
	if w == nn_color:
		score = 1
	elif w == r_engine_color:
		score = 0
	else:
		score = 0.5
		
	return score

########################################################################################################################

""" Training the network """

# The probability to choose a random action, i.e. exploration-exploitation trade-off
epsilon = 0.1
# The discount rate
gamma = 1.0

# The number of episodes, i.e. the number of games
num_episodes = 10000

total_score = 0
score_save_freq = 100
all_scores = []

# Creatig a session
sess = tf.InteractiveSession()
# Initialize the variables
sess.run(tf.global_variables_initializer())
time = { -1 : 0.0, 1 : 0.0 }


for j in range(num_episodes):
	print("Episode #", j)
	total_score = total_score + play_game(-1)
	
	if j % score_save_freq == 0:
		all_scores.append(total_score)
	
	# Mark the players with -1 and 1 according to the color
	# -1 means black color, 1 means white color
	colors = [-1, 1]
	
	# previous state, legal actions and policy action for each agent
	prev_state = {}
	prev_legal_actions = {}
	prev_policy_action = {}
	
	# which player's turn
	turn = 0

	# Initialize the envoronment, i.e. a Board for playing
	board = Board()
	# board.display(time)
	
	# Initial run for both agents
	for c in colors:
		# Read initial state and legal actions
		prev_state[c] = read_state(board)
		prev_legal_actions_idxs = board_to_nn_actions(board.get_legal_moves(c))
		prev_legal_actions_temp = np.zeros(num_output_units)[np.newaxis]
		prev_legal_actions_temp[0][prev_legal_actions_idxs] = 1
		prev_legal_actions[c] = prev_legal_actions_temp
	
	
		# The initial calculating of the Q value
		output_temp = sess.run(a_2_2, feed_dict={X: prev_state[c],
				  	  legal_actions: prev_legal_actions[c]})

		# The initial action associated with the epsilon-greedy policy
		prev_policy_action_temp, _ = epsilon_greedy_policy(epsilon, output_temp,
								 	 prev_legal_actions[c])
		prev_policy_action[c] = prev_policy_action_temp
	
	
		# Now according to the epsilon-greedy policy, make an action and get a new state
		board.execute_move(nn_to_board_action(prev_policy_action[c]), c)
		# board.display(time)
	
	# Until the game lasts, 2 moves made, 58 left out of 60
	for move_num in range(58):
		
		# The game is over
		if not board.get_legal_moves(colors[turn % 2]) and not board.get_legal_moves(colors[(turn + 1) % 2]):	
			# Go to the next episode
			break
		else:
			
			if not board.get_legal_moves(colors[turn % 2]):
				turn = turn + 1
				continue
			
			# Read the state from the board for the current agent
			curr_agent_state = read_state(board)
			# Read the legal actions for the current agent
			curr_agent_legal_actions_idxs = board_to_nn_actions(board.get_legal_moves(colors[turn % 2]))
			curr_agent_legal_actions = np.zeros(num_output_units)[np.newaxis]
			curr_agent_legal_actions[0][curr_agent_legal_actions_idxs] = 1
		
			# Forward pass the current state for the current agent
			curr_agent_output = sess.run(a_2_2, feed_dict={X : curr_agent_state, 
										  legal_actions: curr_agent_legal_actions})
			
			# the action associated with the epsilon-greedy policy q value for the current agent
			curr_agent_policy_action, curr_agent_policy_q = epsilon_greedy_policy(epsilon,
													          curr_agent_output, curr_agent_legal_actions)
																					 	
			# the action associated with the maximal q value for the current agent
			curr_agent_max_action, curr_agent_max_q = greedy_policy(curr_agent_output)
			
			# Forward the current agent previous state with the previous legal actions
			curr_agent_prev_output_again = sess.run(a_2_2, feed_dict={X: prev_state[colors[turn % 2]],
													 legal_actions: prev_legal_actions[colors[turn % 2]]})
			
			# Evaluate a_1 and z_1 layers, in order to feed in the backward pass
			a_1_eval_temp = a_1.eval(feed_dict={X: prev_state[colors[turn % 2]], 
													 legal_actions: prev_legal_actions[colors[turn % 2]]})
			z_1_eval_temp = z_1.eval(feed_dict={X: prev_state[colors[turn % 2]],
													 legal_actions: prev_legal_actions[colors[turn % 2]]})
			z_2_eval_temp = z_2.eval(feed_dict={X: prev_state[colors[turn % 2]],
													 legal_actions: prev_legal_actions[colors[turn % 2]]})				
			
			# Select the q value from the previous run
			curr_agent_prev_policy_q_again = curr_agent_prev_output_again[0, prev_policy_action[colors[turn % 2]]]
			
			# Calculate the new value of the previous q value using the Q-learning formula
			curr_agent_prev_policy_new = 0 + gamma * curr_agent_max_q
			
			# Calculate the error
			error = curr_agent_prev_policy_new - curr_agent_prev_policy_q_again
			
			
			# Make the cost, such that, the only non-zero value will be the error at the index same as the previous action index
			curr_agent_cost = np.zeros(64)[np.newaxis]
			curr_agent_cost[0, prev_policy_action[colors[turn % 2]]] = error
			
			# run the backward pass once
			sess.run(step, feed_dict={X: prev_state[colors[turn % 2]], legal_actions: prev_legal_actions[colors[turn % 2]],
				cost: curr_agent_cost, a_1_eval: a_1_eval_temp, z_1_eval: z_1_eval_temp, z_2_eval: z_2_eval_temp})
			
			
			# Execute the epsilon-greedy action
			board.execute_move(nn_to_board_action(curr_agent_policy_action), colors[turn % 2])
			# board.display(time)
			
			# Update
			prev_state[colors[turn % 2]] = curr_agent_state
			prev_legal_actions[colors[turn % 2]] = curr_agent_legal_actions
			prev_policy_action[colors[turn % 2]] = curr_agent_policy_action
			turn = turn + 1
	
	# After the for loop for the moves
	
	board.display(time)
			
	# Determine the winner
	w = winner(board)
	final_rewards = {}
			
	if w == -1:
		final_rewards[-1] = 1
		final_rewards[1] = 0
	elif w == 1:
		final_rewards[-1] = 0
		final_rewards[1] = 1
	else:
		final_rewards[-1] = 0.5
		final_rewards[1] = 0.5
		
		# last update	
		for c in colors:
			# Calculate the output from the previous state of the current layer
			prev_output_again = sess.run(a_2_2, feed_dict={X: prev_state[c], legal_actions: prev_legal_actions[c]})
				
			# Evaluate a_1 and z_1 layers, in order to feed in the backward pass
			a_1_eval_temp = a_1.eval(feed_dict={X: prev_state[c], legal_actions: prev_legal_actions[c]})
			z_1_eval_temp = z_1.eval(feed_dict={X: prev_state[c], legal_actions: prev_legal_actions[c]})
			z_2_eval_temp = z_2.eval(feed_dict={X: prev_state[c], legal_actions: prev_legal_actions[c]})
				
			# Select the q value with the index of the action from the previous pass, but with updated value
			prev_policy_q_again = prev_output_again[0, prev_policy_action[c]]
				
			# Only the final reward, since it is a terminal state
			prev_policy_new = final_rewards[c]
				
			# Calculate the error
			error = prev_policy_new - prev_policy_q_again
				
			# Make the cost
			cost_temp = np.zeros(64)[np.newaxis]
			cost_temp[0, prev_policy_action[c]] = error
				
			# run the backward pass once
			sess.run(step, feed_dict={X: prev_state[c], legal_actions: prev_legal_actions[c], cost: cost_temp,
		 			a_1_eval: a_1_eval_temp, z_1_eval: z_1_eval_temp, z_2_eval: z_2_eval_temp})		
			

# save the model
saver = tf.train.Saver()
save_path = saver.save(sess, "model/model.ckpt")
print("Model saved in file: %s" % save_path)


# plot the accumulated score during the training
plt.figure()
plt.plot(all_scores)
plt.ylabel("Accumulated score")
plt.xlabel("Number of runs times 100")
# plt.axis([0, num_episodes / score_save_freq, 0, num_episodes])
plt.savefig('score.png')