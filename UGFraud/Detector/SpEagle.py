"""
	'Collective Opinion Spam Detection: Bridging Review Networks and Metadata'
	Utilizing clues from all metadata (text, timestamp, rating) as well as relational data (network),
	and harness them collectively under a unified framework to spot suspicious users and reviews,
	as well as products targeted by spam.
	Article: https://www.andrew.cmu.edu/user/lakoglu/pubs/15-kdd-collectiveopinionspam.pdf
"""

from UGFraud.Utils.helper import *
from heapq import *
from scipy.special import logsumexp
import pickle


class myTuple():
	def __init__(self, cost, node_id):
		self._cost = cost
		self._id = node_id

	def __lt__(self, other):
		return self._cost < other._cost


class Node(object):
	""" a Node object represents a node on the graph (which is also a random variable).

	Attributes:
		_name: node's ID. a string
		_type: a string denoting the type of the node (User, Review, Product)
		_prior: the node's prior distribution (\phi)
		_num_classes: number of classes, which is also the length of the prior vector.
		_outgoing: a dictionary of out-going messages to its neighbors (key: j, value: m_{i\to j})
		where i is the current node and j is the target node.
		_neighbors: a list of references to its neighbors
	"""

	def __init__(self, name, prior, node_type):
		""" Create the attributes
		Args:
			name: a string id of this node.
			prior: a floating number between [0,1] representing P(y=1 | node)
			node_type: 'u', 'p' or 'r'
		"""
		# to prevent log 0
		self._eps = 1e-5

		# node id (such as a u_id, p_id or review_id)
		self._name = name

		# list of names (such as u_id, p_id, and review_id) of the neighboring nodes
		self._neighbors = []

		# a dictionary with key = neighboring node id, value = np.array() representing the message
		# from this node to the neighbor
		self._outgoing = {}

		# prior in log space, with check on 0's
		if prior == 1:
			prior = 1 - self._eps
		elif prior == 0:
			prior = self._eps

		self._prior = np.log(
			np.array([1 - prior, prior]))  # previous version: self._prior = np.log(np.array([1-prior, prior]))

		self._num_classes = 2

		self._type = node_type

	# if self._type == 'p' and self._name == 'p0':
	#	print ('product %s initialized.' % self._name)
	#	print (self._outgoing)

	def add_neighbor(self, neighbor_node_id):
		"""
			add a neighboring node to this node; create out-going message from this node to the neighbor
			Args:
				neighbor_node_id: a string representing the neighbor's id
		"""
		self._neighbors.append(neighbor_node_id)
		self._outgoing[neighbor_node_id] = np.zeros(self._num_classes)

	def add_local_neighbor(self, neighbor_node_id, message):
		"""
			add a neighboring node to this node to build local graph; copy out-going message from the global graph
			Args:
				neighbor_node_id: a string representing the neighbor's id
				message: the message from this node to its neighbor
		"""
		# find the message to the corresponding neighbor_node_id

		for m in message:
			if neighbor_node_id in m.keys():
				message_to_neighbor = m[neighbor_node_id]
				break

		self._neighbors.append(neighbor_node_id)
		self._outgoing[neighbor_node_id] = message_to_neighbor

	def init_outgoing(self):
		"""
			Initialize all messages to 0.
		"""
		# a dictionary: key = neighbor id, value = np.ndarray of uniform distributions
		#self._outgoing = {n: np.zeros(self._num_classes) for n in self._neighbors}
		for n in self._neighbors:
			self._outgoing[n].fill(0.0)

	# if self._type == 'p' and self._name == 'p0':
	#	print (self._outgoing)

	def n_edges(self):
		return len(self._neighbors)

	def get_name(self):
		return self._name

	def get_type(self):
		return self._type

	def get_prior(self):
		"""" return the prior of the node in prob space """
		return np.exp(self._prior)

	def get_neighbors(self):
		return self._neighbors

	def get_outgoing(self):
		return self._outgoing

	def get_message_for(self, neighbor_name):
		""" find the message sent from this node to the neighbor specified by neighbor_name """

		# note that _outgoing is a dictionary with key = neighbor id and value = messages
		# print(neighbor_name)
		assert neighbor_name in self._outgoing, "the neighbor %s is not a neighbor of the node %s\n" % (
		neighbor_name, self._name)

		return self._outgoing[neighbor_name]

	def get_belief(self, all_nodes):
		""" return the belief of the node, along with the messages used to compute the belief
			Args:
				all_nodes: a dictionary containing all nodes on the graph
			Return:
				belief:
				incoming:
		"""

		incoming = []

		# log 1 = 0
		belief = np.zeros(self._num_classes)

		# add log of phi
		belief += self._prior

		# go through each neighbor of the node
		for node_id in self._neighbors:
			# get the message sent from the neighbor n to the current node (self._name)

			# look up the neighboring node in all_nodes
			n = all_nodes[node_id]

			# getting message from the neighboring node to this node
			# consider working in the log scale to prevent underflowing

			# sum log m_ij
			belief += n.get_message_for(self._name)

			# in the same order as self._neighbors
			incoming.append(n.get_message_for(self._name))
		# print (n.get_message_for(self._name))

		return belief, incoming

	def recompute_outgoing(self, potentials, all_nodes, normalize=True):
		""" for each neighbor j, update the message sent to j

			Args:
				potentials: a dictionary (key = edge type, value = log of potential matrix).
					An edge type is src_type + "_" + dst_type
				all_nodes: same as that in get_belief

			Return:
				difference between previous and updated messages.
		"""
		# return value
		diff = 0

		# the messages in incoming is in the same order of self._neighbors
		# total = log phi_i + sum_{j~i} log m_ji
		# incoming = [log m_ji]
		total, incoming = self.get_belief(all_nodes)

		# go through each neighbor of the node
		for j, n_id in enumerate(self._neighbors):

			n = all_nodes[n_id]

			log_m_i = total - incoming[j]

			# note that the potential matrix depends on the edge type (write(user, review) or belong(review, product))
			# edge_type can be (user-review), (review-product), (review-user) and (product-review)
			edge_type = self._type + '_' + n._type

			# log H, where H is symmetric and there is no need to transpose it
			log_H = potentials[edge_type]

			log_m_ij = logsumexp(log_H + np.tile(log_m_i.transpose(), (2, 1)), axis=1)

			# normalize the message
			log_Z = logsumexp(log_H + np.tile(log_m_i.transpose(), (2, 1)))

			log_m_ij -= log_Z#

			# accumulate the difference
			diff += np.sum(np.abs(self._outgoing[n._name] - log_m_ij))

			# set the message from i to j
			self._outgoing[n._name] = log_m_ij
		return diff


class SpEagle:
	def __init__(self, graph, potentials, message=None, max_iters=1):
		""" set up the data and parameters.

		Args:
			graph: a networkx graph

			potentials: a dictionary (key = edge_type, value=np.ndarray)
		"""

		self._potentials = potentials
		self._max_iters = max_iters
		self._message = message

		self._user_priors = node_attr_filter(graph, 'types', 'user', 'prior')
		self._product_priors = node_attr_filter(graph, 'types', 'prod', 'prior')
		self._review_priors = edge_attr_filter(graph, 'types', 'review', 'prior')

		# create nodes on the graph. key = u_id / p_id / review_id, value = node
		self._nodes = {}

		self._bp_schedule = []

		# add nodes and edges to build the graph
		for u_id in self._user_priors.keys():
			unique_u_id = 'u' + u_id

			# prior in log scale
			self._nodes[unique_u_id] = Node(unique_u_id, self._user_priors[u_id], 'u')

			# go through the reviews posted by the user
			for p_id in graph[u_id].keys():
				unique_p_id = 'p' + p_id

				if unique_p_id not in self._nodes:
					self._nodes[unique_p_id] = Node(unique_p_id, self._product_priors[p_id], 'p')

				review_id = (u_id, p_id)
				unique_review_id = (unique_u_id, unique_p_id)

				if unique_review_id not in self._nodes:
					review_node = Node(unique_review_id, self._review_priors[review_id], 'r')

					# add connections and out-going messages if the graph is a global graph
					if self._message is None:
						review_node.add_neighbor(unique_u_id)
						review_node.add_neighbor(unique_p_id)
						self._nodes[unique_u_id].add_neighbor(unique_review_id)
						self._nodes[unique_p_id].add_neighbor(unique_review_id)
					else:
					# add connections and out-going messages if the graph is a local graph
						review_node.add_local_neighbor(unique_u_id, message[unique_review_id])
						review_node.add_local_neighbor(unique_p_id, message[unique_review_id])
						self._nodes[unique_u_id].add_local_neighbor(unique_review_id, message[unique_u_id])
						self._nodes[unique_p_id].add_local_neighbor(unique_review_id, message[unique_p_id])

					self._nodes[unique_review_id] = review_node

	def add_new_data(self, new_user_product_graph, new_priors):
		"""
		Add new a new users-review-products sub-graph to the global existing graph.
		Need to be very careful as we don't want to mess up with existing structures and information

		:param new_user_product_graph: same format as the user_product_graph argument in __init__
		:param new_priors: same format as the priors argument in __init__
		:return: None
		"""
		new_u_priors = new_priors[0]
		new_p_priors = new_priors[2]
		new_r_priors = new_priors[1]

		for u_id, reviews in new_user_product_graph.items():
			unique_u_id = 'u' + u_id
			if unique_u_id not in self._nodes:
				self._user_priors[u_id] = new_u_priors[u_id]
				self._nodes[unique_u_id] = Node(unique_u_id, self._user_priors[u_id], 'u')

			# go through the reviews posted by the user
			for t in reviews:
				p_id = t[0]
				unique_p_id = 'p' + p_id

				if unique_p_id not in self._nodes:
					self._product_priors[p_id] = new_p_priors[p_id]
					self._nodes[unique_p_id] = Node(unique_p_id, self._product_priors[p_id], 'p')

				review_id = (u_id, p_id)
				unique_review_id = (unique_u_id, unique_p_id)

				if unique_review_id not in self._nodes:
					self._review_priors[review_id] = new_r_priors[review_id]
					review_node = Node(unique_review_id, self._review_priors[review_id], 'r')

					# add connections and out-going messages
					review_node.add_neighbor(unique_u_id)
					review_node.add_neighbor(unique_p_id)
					self._nodes[unique_u_id].add_neighbor(unique_review_id)
					self._nodes[unique_p_id].add_neighbor(unique_review_id)
					self._nodes[unique_review_id] = review_node

	def safe_log(self, array, eps=1e-5):
		""" element-wise log the given array with smoothing worrying zeros
		"""
		return np.log((array + eps) / np.sum(array + eps))

	def output_graph(self):
		"""
			output nodes, edges, priors and potentials
		"""
		for n in self._nodes.values():
			print(str(n.get_name()) + ": " + n.get_type())
			print(n.get_prior())
			print(n.get_neighbors())

	def schedule(self, schedule_type='bfs'):
		""" use breadth-first-search to create a BP schedule
		:param:
			schedule_type: 'bfs' or 'degree'
		:return:
		"""

		# sort nodes in descending order of their degrees
		items = [(n.get_name(), n.n_edges()) for k, n in self._nodes.items()]
		items = sorted(items, key=lambda x: x[1], reverse=True)

		if schedule_type == 'degree':
			self._bp_schedule = [name for name, _ in items]
			return

		mark = set(self._nodes.keys())
		self._bp_schedule = []

		head = 0
		tail = -1

		# uncomment this for loop to get bfs + degree
		for node_id, _ in items:
		# uncomment this for loop to get regular bfs
		# for node_id, _ in self._nodes.items():
			node = self._nodes[node_id]
			# newly-found connected component
			if node_id in mark:
				tail += 1
				self._bp_schedule.append(node_id)
				mark.remove(node_id)

				# search starting from i
				while head <= tail:
					cur_node = self._nodes[self._bp_schedule[head]]
					head += 1
					for neighbor_id in cur_node._neighbors:
						if neighbor_id in mark:
							tail += 1
							self._bp_schedule.append(neighbor_id)
							mark.remove(neighbor_id)

	def local_schedule(self, starting_nodes, num_hops):
		"""
		Use Dijkstra to find nodes that are num_hops away from the starting nodes
		:param starting_nodes: the nodes considered the "source"
		:param num_hops: how far away to go
		:return:
		"""
		# node searched so far
		seen = set()
		# minimum distance of each seen node to the source nodes
		min_costs = {}
		# a priority queue of (cost, node_id)
		q = []

		# initialize the queue to contain the starting nodes
		for node_id in starting_nodes:
			q.append(myTuple(0, node_id))
			min_costs[node_id] = 0

		heapify(q)
		self._bp_schedule = []
		while q:
			tuple = heappop(q)
			v = tuple._id
			cost = tuple._cost

			# if the node has cost no greater than num_hops, include it in the update schedule
			if cost <= num_hops:
				self._bp_schedule.append(v)

			if v not in seen:
				# now the node v has its shortest distance to the starting nodes.
				seen.add(v)
				cur_node = self._nodes[v]
				for n in cur_node._neighbors:
					if n not in seen:
						prev = min_costs.get(n, None)
						next = cost + 1
						if prev is None or next < prev:
							min_costs[n] = next
							heappush(q, myTuple(next, n))
		return None

	@timer
	def run_bp(self, start_iter=0, max_iters=-1, early_stop_at=1, tol=1e-3):
		""" run belief propagation on the graph for MaxIters iterations
		Args:
			start_iter: continuing from the results of previous iterations
			max_iters: how many iterations to run BP. Default use the SpEagle's parameter
			early_stop_at: the percentage of nodes whose out-going messages will be updated
			tol: threshold of message differences of one iteration, below which exit BP
		Return:
			delta: the difference in messages before and after iterations of message passing
		"""
		stop_at = int(len(self._bp_schedule) * early_stop_at)

		if max_iters == -1:
			max_iters = self._max_iters

		for it in range(start_iter, start_iter + max_iters, 1):
			if it % 2 == 0:
				start = stop_at - 1
				end = -1
				step = -1
			else:
				start = 0
				end = stop_at
				step = 1
			p = start
			total_updates = 0
			delta = 0
			while p != end:
				total_updates += 1
				cur_node = self._nodes[self._bp_schedule[p]]
				p += step
				delta += cur_node.recompute_outgoing(self._potentials, self._nodes)
				if total_updates > stop_at:
					break
			delta /= total_updates
			# print('bp_iter = %d, delta = %f\n' % (it, delta))
			if abs(delta) < tol:
				break
		return delta

	@timer
	def classify(self):
		""" read out the id of the maximal entry of each belief vector
		Return:
			userBelief: beliefs of the users
			reviewBelief: beliefs of the reviews
			prodBelief: beliefs of the products
		"""
		userBelief= {}
		reviewBelief= {}
		prodBelief= {}

		for k, n in self._nodes.items():
			# decide the type of the node and find its original name
			node_type = None
			if isinstance(k, tuple):
				node_type = 'review'
				u_id = k[0][1:]
				p_id = k[1][1:]
				review_id = (u_id, p_id)
			else:
				if k[0] == 'u':
					node_type = 'user'
					user_id = k[1:]
				else:
					node_type = 'product'
					prod_id = k[1:]

			belief, _ = n.get_belief(self._nodes)

			# from log scale to prob scale and normalize to prob distribution
			posterior_med = np.exp(belief)
			posterior = posterior_med / np.sum(posterior_med)

			if node_type == 'review':
				reviewBelief[review_id] = posterior[1]
			elif node_type == 'user':
				userBelief[user_id] = posterior[1]
			elif node_type == 'product':
				prodBelief[prod_id] = posterior[1]
			else:
				continue

		return userBelief, reviewBelief, prodBelief


if __name__ == '__main__':
	prefix = '/Users/dozee/Desktop/Reseach/Spam_Detection/Dataset/YelpChi/'
	metadata_filename = prefix + 'metadata.gz'

	# prior file names
	user_prior_filename = prefix + 'UserPriors.pickle'
	prod_prior_filename = prefix + 'ProdPriors.pickle'
	review_prior_filename = prefix + 'ReviewPriors.pickle'

	# read the graph and node priors
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)

	with open(user_prior_filename, 'rb') as f:
		user_priors = pickle.load(f)

	with open(prod_prior_filename, 'rb') as f:
		prod_priors = pickle.load(f)

	with open(review_prior_filename, 'rb') as f:
		review_priors = pickle.load(f)

	# print(user_priors)
	# set up edge potentials
	'''
	User and Review potential
		[1,0]
		[0,1]
	Reviewer and Review potential
		[1 - eps, eps]
		[eps, 1 - eps]
	'''
	numerical_eps = 1e-5
	user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
	eps = 0.1
	review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))

	potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
				  'r_p': review_product_potential, 'p_r': review_product_potential}

	model = SpEagle(user_product_graph, [user_priors, prod_priors, review_priors], potentials, max_iters=100)
	model.schedule()
	model.run_bp()

