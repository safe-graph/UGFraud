from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import gzip
import numpy as np


def create_ground_truth(user_data):
	"""Given user data, return a dictionary of labels of users and reviews
	Args:
		user_data: key = user_id, value = list of review tuples.
	Return:
		user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
		review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
	"""
	user_ground_truth = {}
	review_ground_truth = {}

	for user_id, reviews in user_data.items():

		user_ground_truth[user_id] = 0

		for r in reviews:
			prod_id = r[0]
			label = r[2]

			if label == -1:
				review_ground_truth[(user_id, prod_id)] = 1
				user_ground_truth[user_id] = 1
			else:
				review_ground_truth[(user_id, prod_id)] = 0

	return user_ground_truth, review_ground_truth


def evaluate(y, pred_y):
	"""
	Revise: test when a key is a review/account.
	Evaluate the prediction of account and review by SpEagle
	Args:
		y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)
		pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
				the keys in pred_y must be a subset of the keys in y
	"""
	posteriors = []
	ground_truth = []

	for k, v in pred_y.items():
		if k in y:
			posteriors.append(v)
			ground_truth.append(y[k])


	auc = roc_auc_score(ground_truth, posteriors)
	ap = average_precision_score(ground_truth, posteriors)

	return auc, ap


def scale_value(value_dict):
	'''
	Calculate and return a dict of the value of input dict scaled to (0, 1)
	'''
	minimum_value = 0

	ranked_dict = [(user, value_dict[user]) for user in value_dict.keys()]
	ranked_dict = sorted(ranked_dict, reverse=True, key=lambda x: x[1])

	up_max, up_mean, up_min = ranked_dict[0][1], ranked_dict[int(len(ranked_dict) / 2)][1], ranked_dict[-1][1]

	scale_dict = {}
	for i, p in value_dict.items():
		norm_value = (p - up_min) / (up_max - up_min)

		if norm_value == 0:  # avoid the 0
			scale_dict[i] = 0 + 1e-7
		elif norm_value == 1:  # avoid the 1
			scale_dict[i] = 1 - 1e-7
		else:
			scale_dict[i] = norm_value

	return scale_dict


def nor_priors(priors):
	"""
	Normalize the node priors for GANG
	:param priors:
	:return:
	"""
	new_upriors, new_rpriors, new_ppriors = priors

	# normalize the node priors to (0,1)
	# if we normalize the prior, we need to set nor_flg to True for the gang model
	ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
	ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
	ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
	ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
	ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
	u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
	p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
	r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]
	for i, p in priors[0].items():
		priors[0][i] = (p - u_min) / (u_max - u_min)
	for i, p in priors[1].items():
		priors[1][i] = (p - r_min) / (r_max - r_min)
	for i, p in priors[2].items():
		priors[2][i] = (p - p_min) / (p_max - p_min)

	return priors, [u_mean, r_mean, p_mean]


def read_graph_data(metadata_filename, adj=False):
	""" Read the user-review-product graph from file. Can output the graph in different formats
		Args:
			metadata_filename: a gzipped file containing the graph.
			adj: if True: create adjacent data, default is False
		Return:
			graph: user-review / prod-review / list of adjacent(adj=True)
	"""

	user_data = {}

	prod_data = {}

	adj_data = []

	# use the rt mode to read ascii strings instead of binary
	if adj is False:
		with gzip.open(metadata_filename, 'rt') as f:
			# file format: each line is a tuple (user id, product id, rating, label, date)
			for line in f:
				items = line.strip().split()
				u_id = items[0]
				p_id = items[1]
				rating = float(items[2])
				label = int(items[3])
				date = items[4]

				if u_id not in user_data:
					user_data[u_id] = []
				user_data[u_id].append((p_id, rating, label, date))

				if p_id not in prod_data:
					prod_data[p_id] = []
				prod_data[p_id].append((u_id, rating, label, date))

				# create adj_list [u_id, p_id, 1/2], where 1 indicates positive rating (4, 5)
				# and 2 indicates negative rating (1, 2, 3)

		print('read reviews from %s' % metadata_filename)
		print('number of users = %d' % len(user_data))
		print('number of products = %d' % len(prod_data))
		return user_data, prod_data
	else:
		# create adj_list [u_id, p_id, 1/2], where 1 indicates positive rating (4, 5)
		# and 2 indicates negative rating (1, 2, 3)
		with gzip.open(metadata_filename, 'rt') as f:
			# file format: each line is a tuple (user id, product id, rating, label, date)
			for line in f:
				items = line.strip().split()
				u_id = items[0]
				p_id = items[1]
				rating = float(items[2])
				label = int(items[3])
				date = items[4]

				if u_id not in user_data:
					user_data[u_id] = []
				user_data[u_id].append((p_id, rating, label, date))

				if p_id not in prod_data:
					prod_data[p_id] = []
				prod_data[p_id].append((u_id, rating, label, date))

				if int(rating) <= 3:
					rating = int(2)
				else:
					rating = int(1)
				adj_data.append([u_id, p_id, rating])

		print('read reviews from %s' % metadata_filename)
		print('number of users = %d' % len(user_data))
		print('number of products = %d' % len(prod_data))
		print('number of ratings = %d' % len(adj_data))
		return user_data, prod_data, np.array(adj_data, dtype='int32')

