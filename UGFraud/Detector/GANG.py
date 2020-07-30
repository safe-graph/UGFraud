"""
	'GANG: Detecting Fraudulent Users in Online Social Networks via Guilt-by-Association on Directed Graphs'
	A guilt-by-association method on directed graphs, to detect fraudulent users in OSNs.
	Article: http://people.duke.edu/~zg70/papers/GANG.pdf
"""

from scipy.sparse import lil_matrix
from UGFraud.Utils.helper import *
import random


def semi_data(ground_truth, portion):
	"""
	produce the sampled labeled review id used for semi-supervised prior
	:param ground_truth: dict of ground truth {uid:label} or {rid:label}
	:param portion: portion of the labeled data
	:return: review id which are used for supervising
	"""

	smaple_size = int(len(ground_truth) * portion * 0.5)
	total_list = [r for r in ground_truth.keys()]
	pos_list = []
	neg_list = []
	for id, label in ground_truth.items():
		if label == 1:
			pos_list.append(id)
		else:
			neg_list.append(id)

	pos_sample = [pos_list[i] for i in sorted(random.sample(range(len(pos_list)), smaple_size))]
	neg_sample = [neg_list[i] for i in sorted(random.sample(range(len(neg_list)), smaple_size))]

	pos_ids = [total_list.index(s) for s in pos_sample]
	neg_ids = [total_list.index(s) for s in neg_sample]

	return pos_ids, neg_ids


class GANG:

	def __init__(self, graph, user_ground_truth, sup_per, nor_flg, sup_flg=False):

		# number of dimensions of product-user matrix
		u_prior = node_attr_filter(graph, 'types', 'user', 'prior')
		p_prior = node_attr_filter(graph, 'types', 'prod', 'prior')
		r_prior = edge_attr_filter(graph, 'types', 'review', 'prior')
		priors = [u_prior, r_prior, p_prior]
		self.pu_dim = len(priors[0])+len(priors[2])
		# spam belief prior vector
		self.res_pu_spam_prior_vector = None
		# diagonal matrix used for normalization
		self.diag_pu_matrix = None
		# product-user spam posterior belief vector
		self.res_pu_spam_post_vector = np.zeros((self.pu_dim, 1))
		# sparse row matrix is faster when multiply with vectors
		self.pu_csr_matrix = None
		self.diag_pu_csr_matrix = None
		self.nor_pu_csr_matrix = None
		# priors dictionary
		self.u_priors = priors[0]
		self.r_priors = priors[1]
		self.p_priors = priors[2]
		# build prior belief vector
		p_vector, u_vector, r_vector = [], [], []
		if nor_flg:
			# the mean value with normalization
			u_mean, p_mean, r_mean = 0.5, 0.5, 0.5
		else:
			# the mean value without normalization
			priors, mean_priors = nor_priors(priors)
			u_mean, r_mean, p_mean = mean_priors[0], mean_priors[1], mean_priors[2]

		for u in priors[0].values():
			u_vector.append(u)
		for p in priors[2].values():
			p_vector.append(p)

		res_u_vector = [i-u_mean for i in u_vector]
		res_p_vector = [i-p_mean for i in p_vector]

		# add semi-supervised user information
		if sup_flg:
			pos_ids, neg_ids = semi_data(user_ground_truth, sup_per)
			for iter, prob in enumerate(res_u_vector):
				if iter in pos_ids:
					res_u_vector[iter] = 1 - u_mean
				elif iter in neg_ids:
					res_u_vector[iter] = 0 - u_mean

		# aggregate the prior vectors
		res_pu_vector = res_p_vector + res_u_vector

		self.res_pu_spam_prior_vector = np.c_[res_pu_vector]

		# build product-user adjacency sparse matrix
		self.pu_matrix = lil_matrix((self.pu_dim, self.pu_dim))

		# create the pu diagonal matrix
		self.diag_pu_matrix = lil_matrix((self.pu_dim, self.pu_dim))
		for id in range(0, self.pu_dim):
			if id < len(self.p_priors):
				self.diag_pu_matrix[id, id] = len(graph[str(id)])
			else:
				self.diag_pu_matrix[id, id] = len(graph[str(id)])

		for p_id in p_prior.keys():
			for neighbor_id in graph[p_id].keys():
				self.pu_matrix[int(p_id), int(neighbor_id)] = 1

		for u_id in u_prior.keys():
			for neighbor_id in graph[u_id].keys():
				self.pu_matrix[int(u_id), int(neighbor_id)] = 1

	@timer
	def pu_lbp(self, max_iters):
		"""
		Run the matrix form of lbp on the product-user sparse matrix
		:return: the posterior belief vector of products and users
		"""

		# transfer to sparse row matrix to accelerate calculation
		self.pu_csr_matrix = self.pu_matrix.tocsr()
		self.diag_pu_csr_matrix = self.diag_pu_matrix.tocsr()

		i = 0
		while i < max_iters:
			sum_0 = np.sum(self.res_pu_spam_post_vector)
			self.res_pu_spam_post_vector = self.res_pu_spam_prior_vector + 2 * 0.008 * (self.pu_csr_matrix.dot(self.res_pu_spam_post_vector))
			sum_1 = np.sum(self.res_pu_spam_post_vector)

			# print('iter: ' + str(i))
			# print('diff: ' + str(abs(sum_0 - sum_1)))

			i += 1

			if abs(sum_0 - sum_1) < 0.1:
				return abs(sum_0 - sum_1)

	@timer
	def classify(self):
		"""
		Calculate the posterior belief of three type of nodes
		:return: u_post: users posterior beliefs, p_post: products posterior beliefs,
		r_post: reviews posterior beliefs.
		"""
		u_post = {}
		p_post = {}
		r_post = {}
		pu_post = self.res_pu_spam_post_vector
		no_prod = len(self.p_priors)
		# extract the posterior belief of users and reviews
		for i, r in enumerate(pu_post[no_prod:]):
			u_post[str(i + no_prod)] = float(r)
		for i, r in enumerate(pu_post[:no_prod]):
			p_post[str(i)] = float(r)
		for i, r in self.r_priors.items():
			r_post[i] = (u_post[i[0]] + float(r)) / 2

		u_post = scale_value(u_post)
		p_post = scale_value(p_post)
		r_post = scale_value(r_post)

		return u_post, p_post, r_post