import copy as cp
import random as rd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from Utils.yelpFeatureExtraction import *
from Utils.iohelper import *

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


def create_ground_truth_with_labeled_reviews(user_data, labeled_reviews):
	"""Given user data, return a dictionary of labels of users and reviews
	Args:
		user_data: key = user_id, value = list of review tuples.
		labeled_reviews: key = review_id, value = 1 (spam) / 0 (non-spam)
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

# skip labeled ones
			if (user_id, prod_id) in labeled_reviews:
				continue

			if label == -1:
				review_ground_truth[(user_id, prod_id)] = 1
				user_ground_truth[user_id] = 1
			else:
				review_ground_truth[(user_id, prod_id)] = 0

	return user_ground_truth, review_ground_truth


def create_evasion_ground_truth(user_data, evasive_spams):
	"""Assign label 1 to evasive spams and 0 to all existing reviews; Assign labels to accounts accordingly
	Args:
		user_data: key = user_id, value = list of review tuples.
			user_data can contain only a subset of reviews
			(for example, if some of the reviews are used for training)

		evasive_spams: key = product_id, value = list of review tuples

	Return:
		user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
		review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
	"""
	old_spammers = set()
	old_spams = set()

	user_ground_truth = {}
	review_ground_truth = {}

	# assign label 0 to all existing reviews and users
	for user_id, reviews in user_data.items():
		user_ground_truth[user_id] = 0

		for r in reviews:
			prod_id = r[0]
			label = r[2]
			review_ground_truth[(user_id, prod_id)] = 0

			if label == -1:
				old_spams.add((user_id, prod_id))
				old_spammers.add(user_id)

	# exclude previous spams and spammers, since the controlled accounts are selcted from the normal accounts.
	for r_id in old_spams:
		review_ground_truth.pop(r_id)
	for u_id in old_spammers:
		user_ground_truth.pop(u_id)

	# add label 1 to the evasive spams
	for prod_id, spams in evasive_spams.items():

		for r in spams:
			user_id = r[0]

			review_ground_truth[(user_id, prod_id)] = 1
			# this user now has posted at least one spam, so set its label to 1
			user_ground_truth[user_id] = 1
	# print(evasive_spams)
	# print(len(user_ground_truth))
	return user_ground_truth, review_ground_truth


def create_review_features(userFeatures, prodFeatures, reviewFeatures, userFeatureNames, prodFeatureNames, reviewFeatureNames):
	"""
	Concatenate product and user features to each review, as the review's features
	:param userFeatures:
	:param prodFeatures:
	:param reviewFeatures:
	:return:
	"""
	review_mat = []
	for r, rf in reviewFeatures.items():
		u_id = r[0]
		p_id = r[1]
		uf = userFeatures[u_id]
		pf = prodFeatures[p_id]

		review_feature_vector = []
		for fn in reviewFeatureNames:
			if fn in rf:
				review_feature_vector.append(rf[fn])
			else:
				review_feature_vector.append(np.inf)

		for fn in prodFeatureNames:
			if fn in pf:
				review_feature_vector.append(pf[fn])
			else:
				review_feature_vector.append(np.inf)

		for fn in userFeatureNames:
			if fn in uf:
				review_feature_vector.append(uf[fn])
			else:
				review_feature_vector.append(np.inf)
		review_mat.append(review_feature_vector)

	review_mat = np.array(review_mat)
	for col in range(review_mat.shape[1]):
		non_inf = np.logical_not(np.isinf(review_mat[:, col]))
		m = np.mean(review_mat[non_inf, col])
		# replace inf with mean
		review_mat[np.isinf(review_mat[:, col]), col] = m

	review_feature_dict = {}
	i = 0
	for r, _ in reviewFeatures.items():
		review_feature_dict[r] = review_mat[i,:]
		i+=1
	return review_feature_dict, review_mat.shape[1]


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

	#     print ('number of test reviews: %d' % len(review_ground_truth))
	#     print ('number of test users: %d' % len(user_ground_truth))

	auc = roc_auc_score(ground_truth, posteriors)
	ap = average_precision_score(ground_truth, posteriors)

	return auc, ap


def roc(y, pred_y):
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

	#     print ('number of test reviews: %d' % len(review_ground_truth))
	#     print ('number of test users: %d' % len(user_ground_truth))

	fpr, tpr, threshold = roc_curve(ground_truth, posteriors)

	return fpr, tpr, threshold


def precision_recall(y, pred_y):
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

	#     print ('number of test reviews: %d' % len(review_ground_truth))
	#     print ('number of test users: %d' % len(user_ground_truth))

	precision, recall, threshold = precision_recall_curve(ground_truth, posteriors)

	return precision, recall, threshold


def precision_top_k(y, pred_y, k):
	"""
	Compute the top-k precision, along with the top k items with their true labels.
	Args:
		y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

		pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
				the keys in pred_y must be a subset of the keys in y
	Return:
		topK precision
		items in the top k list
	"""
	sorted_list = sorted([(k,v) for k,v in pred_y.items()], key = lambda x:x[1])
	top_k_items = [k for k,v in sorted_list[:k]]
	top_k_labels = [y[k] for k in top_k_items]
	return float(sum(top_k_labels)) / k, top_k_items


def sample_labeled_reviews(user_data, percentage):
	"""Sample some reviews as labeled data
	Note that the user_id and product_id may duplicate: there is no u or p prefix to them.
	"""
	spams = []
	non_spams = []
	for user_id, reviews in user_data.items():
		for r in reviews:
			label = r[2]
			prod_id = r[0]

			if label == -1:
				spams.append((user_id, prod_id))
			else:
				non_spams.append((user_id, prod_id))

	idx = np.random.choice(len(spams), int(len(spams) * percentage), replace=False)
	labeled_spams = [spams[i] for i in idx]
	idx = np.random.choice(len(non_spams), int(len(non_spams) * percentage), replace=False)
	labeled_non_spams = [non_spams[i] for i in idx]

	return labeled_spams, labeled_non_spams


def reset_priors_with_labels(priors, node_labels):
	"""Set node priors (of a single type: user, product or review) according to the given node labels
	Args:
		priors: original node priors
		node_labels: a dictionary with key = node id and value = label (0 non-spam, 1 spam)
	"""
	for node_id, label in node_labels.items():
		assert (node_id in priors), 'Review %s not in priors' % node_id
		if label == 1:
			priors[node_id] = 0.999
		elif label == 0:
			priors[node_id] = 0.001


def reset_priors_with_priors(priors, node_priors):
	"""Set node priors (of a single type: user, product or review) according to the given node priors
	Args:
		priors: original node priors
		node_priors: a dictionary with key = node id and value = p(y=spam|node)
	"""
	for node_id, label in node_priors.items():
		assert (node_id in priors), 'Review %s not in priors' % node_id
		priors[node_id] = node_priors[node_id]


def practical_metric(threshold, meta_path, spam_path, new_user_product_graph, new_product_user_graph, review_ground_truth, reviewBelief, attack_para, elite):
	'''
	Calculate the revenue change after filtering the detected reviews
	:param threshold: The top % of reviews checked
	:param meta_path: The original graph file path
	:param spam_path: The added edges file path
	:param new_product_user_graph: The new graph with injected edges
	:param reviewBelief: The review posterior beliefs of the new graph
	:return: (original revenue, revenue change)
	'''
	thresholds = np.arange(0, 1, 0.05)
	metadata_filename = meta_path + 'metadata.gz'
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)
	with open(spam_path, 'rb') as f:
		spammers, targets, added_edges = pickle.load(f)

	pos = sum(review_ground_truth.values())
	plot_x = []
	plot_y = []
	# calculate the original revenue
	original_avg_ratings = {}
	for product, reviews in product_user_graph.items():
		rating = 0
		for review in reviews:
			rating += review[1]
		original_avg_ratings[product] = rating/len(reviews)

	original_mean_rating = sum(r for r in original_avg_ratings.values())/len(original_avg_ratings)
	original_RD = {}
	original_EAR = {}
	original_Revenue = {}
	for target in targets:
		original_RD[target] = original_avg_ratings[target] - original_mean_rating
		temp_EAR = []
		for review in product_user_graph[target]:
			if len(user_product_graph[review[0]]) >= elite:
				temp_EAR.append(review[1])
		original_EAR[target] = sum(temp_EAR)/len(temp_EAR) if len(temp_EAR) != 0 else 0

	for target in targets:
		original_Revenue[target] = 0.09 + 0.035 * original_RD[target] + 0.036 * original_EAR[target]

	for threshold in thresholds:
		# for each filtering threshold, we calculate the promotion value and F1
		# filter detected non-singleton reviews:
		ranked_rpost = [(review, float(reviewBelief[review])) for review in reviewBelief.keys()]
		ranked_rpost = sorted(ranked_rpost, reverse=True, key=lambda x: x[1])
		filter_reviews = []
		for rpost in ranked_rpost[:int(len(ranked_rpost)*threshold)]:
			filter_reviews.append(rpost[0])
		TP = 0
		# remove the detected reviews under the mix attack
		if len(spammers) == attack_para[0]:

			for edge in added_edges:
				if edge in filter_reviews:
					TP += 1
					for review in new_product_user_graph[edge[1]]:
						if review[0] == edge[0]:
							new_product_user_graph[edge[1]].remove(review)
					for review in new_user_product_graph[edge[0]]:
						if review[0] == edge[1]:
							new_user_product_graph[edge[0]].remove(review)

			for iter, target in enumerate(targets):
				for user in spammers[attack_para[2]+iter*15:attack_para[2]+iter*15+15]:
					edge = (user, target)
					if edge in filter_reviews:
						TP += 1
						for review in new_product_user_graph[edge[1]]:
							if review[0] == edge[0]:
								new_product_user_graph[edge[1]].remove(review)
						for review in new_user_product_graph[edge[0]]:
							if review[0] == edge[1]:
								new_user_product_graph[edge[0]].remove(review)

		# remove detected singleton reviews if under singleton attack
		if len(added_edges) == 0:
			for iter, target in enumerate(targets):
				for user in spammers[iter*30:iter*30+30]:
					edge = (user, target)
					if edge in filter_reviews:
						TP += 1
						for review in new_product_user_graph[edge[1]]:
							if review[0] == edge[0]:
								new_product_user_graph[edge[1]].remove(review)
						for review in new_user_product_graph[edge[0]]:
							if review[0] == edge[1]:
								new_user_product_graph[edge[0]].remove(review)

		# calculate the new revenue
		new_avg_ratings = {}
		for product, reviews in new_product_user_graph.items():
			rating = 0
			for review in reviews:
				rating += review[1]
			new_avg_ratings[product] = rating/len(reviews)

		new_mean_rating = sum(r for r in new_avg_ratings.values())/len(new_avg_ratings)
		new_RD = {}
		new_EAR = {}
		new_Revenue = {}
		for target in targets:
			new_RD[target] = new_avg_ratings[target] - new_mean_rating
			temp_EAR = []
			for review in new_product_user_graph[target]:
				if len(new_user_product_graph[review[0]]) >= elite:
					temp_EAR.append(review[1])
			new_EAR[target] = sum(temp_EAR)/len(temp_EAR) if len(temp_EAR) != 0 else 0

		differences = {}
		for target in targets:
			new_Revenue[target] = 0.09 + 0.035 * new_RD[target] + 0.036 * new_EAR[target]
			differences[target] = new_Revenue[target] - original_Revenue[target]

		delta_R = sum(differences.values()) # / len(differences)

		fscore = 2*TP/(2*TP + pos-TP)

		# print(delta_R)
		# print(fscore)
		# print('----')
		#
		# for target in targets:
		# 	print(original_avg_ratings[target])
		# 	print(new_avg_ratings[target])
		# 	print(original_RD[target])
		# 	print(new_RD[target])
		# 	print(original_EAR[target])
		# 	print(new_EAR[target])
		# 	print(original_Revenue[target])
		# 	print(new_Revenue[target])
		# 	print(len(product_user_graph[target]))
		# 	print('-')
		# exit()

		plot_x.append(fscore)
		plot_y.append(delta_R)


	return original_Revenue, [plot_x, plot_y]


def randomDate(origin_date, range):
	"""
	The function generate a random date with a given date
	:param origin_date:
	:return: the new random date
	"""
	even_month = [4, 6, 9, 11]
	odd_month = [1, 3, 5, 7, 8, 10, 12]

	# month_range = rd.randint(0, 6)
	day = int(origin_date[8:])
	month = int(origin_date[5:7])
	year = int(origin_date[:4])
	new_day = day + rd.randint(0, range)
	if new_day > 28 and month == 2:
		month += 1
		new_day -= 28
	elif new_day > 30 and month in even_month:
		month += 1
		new_day -= 30
	elif new_day > 31 and month in odd_month:
		month += 1
		new_day -= 31
	if month > 12:
		year += 1
		month = 1
	# month = int(origin_date[5:6])  + month_range
	# new_month = '0' + str(month % 12 + 1) if 0 <= month % 12 < 9 else str(month % 12 + 1)
	new_month = '0' + str(month) if month < 10 else str(month)
	new_day = '0' + str(new_day) if new_day < 10 else str(new_day)
	new_date = str(year) + '-' + new_month + '-' + new_day
	# print(new_date)
	return new_date


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

		if norm_value == 0: # avoid the 0
			scale_dict[i] = 0 + 1e-7
			# for j in range(2, len(value_dict)):
			# 	if ranked_dict[-j][1] != up_min:
			# 		scale_dict[i] = (ranked_dict[-j][1] - up_min) / (up_max - up_min)
			# 		minimum_value = scale_dict[i]
			# 		break
		elif norm_value == 1: # avoid the 1
			scale_dict[i] = 1 - 1e-7
		else:
			scale_dict[i] = norm_value

	return scale_dict


def normalize(dict):
	"""
	normalized given dictionary value to [0,1]
	"""

	total = sum([v for v in dict.values()])

	for i, v in dict.items():
		dict[i] = v/total

	return dict


def ranking_metric(meta_path, spam_path, new_user_product_graph, new_product_user_graph, review_ground_truth, reviewBelief, attack_para, elite, top_k):

	metadata_filename = meta_path + 'metadata.gz'
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)
	with open(spam_path, 'rb') as f:
		spammers, targets, added_edges = pickle.load(f)

	pos = sum(review_ground_truth.values())
	plot_x = []
	plot_y = []

	# calculate the original revenue
	original_avg_ratings = {}
	for product, reviews in product_user_graph.items():
		rating = 0
		for review in reviews:
			rating += review[1]
		original_avg_ratings[product] = rating/len(reviews)

	original_mean_rating = sum(r for r in original_avg_ratings.values())/len(original_avg_ratings)
	original_RD = {}
	original_EAR = {}
	original_Revenue = {}
	for target in targets:
		original_RD[target] = original_avg_ratings[target] - original_mean_rating
		temp_EAR = []
		for review in product_user_graph[target]:
			if len(user_product_graph[review[0]]) >= elite:
				temp_EAR.append(review[1])
		original_EAR[target] = sum(temp_EAR)/len(temp_EAR) if len(temp_EAR) != 0 else 0

	for target in targets:
		original_Revenue[target] = 0.09 + 0.035 * original_RD[target] + 0.036 * original_EAR[target]

	ranked_rpost = [(review, float(reviewBelief[review])) for review in reviewBelief.keys()]
	ranked_rpost = sorted(ranked_rpost, reverse=True, key=lambda x: x[1])

	top_k_list = ranked_rpost[:top_k]

	top_k_label = []
	for k in top_k_list:
		if k[0] in review_ground_truth.keys(): # and len(new_user_product_graph[k[0][0]])>1:
			top_k_label.append(review_ground_truth[k[0]])
		else:
			top_k_label.append(0)

	return top_k_list, top_k_label


def add_fake_reviews(upg, pug, new_edges, spammer_ids, target_ids, attack_para):
	"""
	add fake reviews to all reviews and create new ground truth
	:param added_review:
	:return:
	"""

	#### Promotion or demotion setting
	rating = 5
	####
	user_product_graph = cp.deepcopy(upg)
	prod_user_graph = cp.deepcopy(pug)
	new_user_graph = {}
	new_product_graph = {}

	feature_suspicious_filename = 'feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

	# read the graph and node priors
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()


	if len(new_edges) != 0 and len(spammer_ids) == attack_para[0]:
		# we are under mix attack setting, so we need to load the new added edges and update the features

		# load evasive edges to a new graph
		for added_edge in new_edges:
			added_account = added_edge[0]
			target = added_edge[1]
			origin_date = user_product_graph[added_account][0][3]
			new_date = origin_date  # randomDate(origin_date)
			new_rating = rating  # rd.randint(4, 5)
			if added_account not in new_user_graph.keys():
				# a tuple of (product_id, rating, label, posting_date)
				new_user_graph[added_account] = [(target, new_rating, -1, new_date)]
			else:
				new_user_graph[added_account].append((target, new_rating, -1, new_date))
			if target not in new_product_graph.keys():
				# a tuple of (user_id, rating, label, posting_date)
				new_product_graph[target] = [(added_account, new_rating, -1, new_date)]
			else:
				new_product_graph[target].append((added_account, new_rating, -1, new_date))
		# add the singleton reviews to complete graph
		origin_date = '2012-06-01'
		new_date = origin_date  # randomDate(origin_date)
		new_rating = rating  # rd.randint(4, 5)
		for iter, target in enumerate(target_ids):
			for user in spammer_ids[attack_para[2] + iter * 15:attack_para[2] + iter * 15 + 15]:
				added_account = user
				user_product_graph[added_account] = [(target, new_rating, -1, new_date)]
				prod_user_graph[target].append((added_account, new_rating, -1, new_date))
		# calculate feature_extractorres on the complete graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																							  prod_user_graph)
		# update features with the new graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph,
																						   new_user_graph,
																						   prod_user_graph,
																						   new_product_graph,
																						   UserFeatures,
																						   ProdFeatures, ReviewFeatures)
	else:  # we are under the singleton attack setting
		# we need to add the new nodes and edges to the complete graph
		origin_date = '2012-06-01'
		new_date = origin_date  # randomDate(origin_date)
		new_rating = rating  # rd.randint(4, 5)
		for iter, target in enumerate(target_ids):
			for user in spammer_ids[iter * 30:iter * 30 + 30]:
				added_account = user
				# add the new review to the complete graph
				user_product_graph[added_account] = [(target, new_rating, -1, new_date)]
				prod_user_graph[target].append((added_account, new_rating, -1, new_date))

		# calculate features on the complete graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																							  prod_user_graph)

	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

	user_priors = new_upriors
	review_priors = cp.deepcopy(new_rpriors)
	prod_priors = new_ppriors

	# create ground truth under mix attack
	evasive_spams = {}
	if len(new_edges) == attack_para[1]:
		for added_edge in new_edges:
			added_account = added_edge[0]
			target = added_edge[1]
			if target not in evasive_spams.keys():
				evasive_spams[target] = [(added_account, rating, -1, '2012-06-01')]
			else:
				evasive_spams[target].append((added_account, rating, -1, '2012-06-01'))
		for iter, target in enumerate(target_ids):
			for user in spammer_ids[attack_para[2] + iter * 15:attack_para[2] + iter * 15 + 15]:
				if target not in evasive_spams.keys():
					evasive_spams[target] = [(user, rating, -1, '2012-06-01')]
				else:
					evasive_spams[target].append((user, rating, -1, '2012-06-01'))

	# if we are under the singleton attack setting, we modify their original posts as new spam review
	if len(new_edges) == 0:
		for iter, target in enumerate(target_ids):
			for user in spammer_ids[iter * 30:iter * 30 + 30]:
				if target not in evasive_spams.keys():
					evasive_spams[target] = [(user, rating, -1, '2012-06-01')]
				else:
					evasive_spams[target].append((user, rating, -1, '2012-06-01'))

	# add new edge into graph
	# add new edges into the original graph
	for e in new_edges:
		u_id = str(e[0])
		p_id = str(e[1])
		user_product_graph[u_id].append((p_id, rating, -1, '2012-06-01'))
		prod_user_graph[p_id].append((u_id, rating, -1, '2012-06-01'))

	user_ground_truth, review_ground_truth = create_evasion_ground_truth(user_product_graph, evasive_spams)

	return [user_priors, review_priors, prod_priors, new_upriors, new_rpriors, new_ppriors], user_product_graph, prod_user_graph, user_ground_truth, review_ground_truth


def add_adversarial_review(user_product_graph, prod_user_graph, new_edges):
	"""
	Add fake reviews for adversarial training
	"""

	#### Promotion or demotion setting
	rating = 5
	####

	new_user_graph = {}
	new_product_graph = {}
	single_mapping = {}

	feature_suspicious_filename = 'feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

	# read the graph and node priors
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()

	# load evasive edges to a new graph to accelerate the prior computation
	for added_edge in new_edges:
		added_account = added_edge[0]
		target = added_edge[1]
		# add elite reviews
		if added_account in user_product_graph.keys():
			origin_date = user_product_graph[added_account][0][3]
			new_date = origin_date #randomDate(origin_date, 5)
			new_rating = rating #rd.randint(4, 5)
			if added_account not in new_user_graph.keys():
				# a tuple of (product_id, rating, label, posting_date)
				new_user_graph[added_account] = [(target, new_rating, -1, new_date)]
			else:
				new_user_graph[added_account].append((target, new_rating, -1, new_date))
			if target not in new_product_graph.keys():
				# a tuple of (user_id, rating, label, posting_date)
				new_product_graph[target] = [(added_account, new_rating, -1, new_date)]
			else:
				new_product_graph[target].append((added_account, new_rating, -1, new_date))
		else:
			# add singleton reviews
			origin_date = '2012-06-01'
			new_date = origin_date #randomDate(origin_date, 5)
			new_rating = rating #rd.randint(4, 5)
			new_account = str(len(user_product_graph) + len(prod_user_graph))
			single_mapping[added_account] = new_account
			user_product_graph[new_account] = [(target, new_rating, -1, new_date)]
			prod_user_graph[target].append((new_account, new_rating, -1, new_date))


	# calculate feature_extractorres on the complete graph
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																						  prod_user_graph)


	if len(new_user_graph) != 0:
		# update features with the new graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph,
																						   new_user_graph,
																						   prod_user_graph,
																						   new_product_graph,
																						   UserFeatures,
																						   ProdFeatures, ReviewFeatures)

	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

	user_priors = new_upriors
	review_priors = cp.deepcopy(new_rpriors)
	prod_priors = new_ppriors

	# create evasion reviews set
	evasive_spams = {}
	new_added_reviews = []
	for added_edge in new_edges:
		added_account = added_edge[0]
		target = added_edge[1]
		if added_account in single_mapping.keys():
			added_account = single_mapping[added_account]
		if target not in evasive_spams.keys():
			evasive_spams[target] = [(added_account, rating, -1, '2012-06-01')]
		else:
			evasive_spams[target].append((added_account, rating, -1, '2012-06-01'))

		new_added_reviews.append((added_account, target))


	# add new edges into the original graph
	for e in new_edges:
		u_id = str(e[0])
		p_id = str(e[1])
		if u_id not in single_mapping.keys():
			user_product_graph[u_id].append((p_id, rating, -1, '2012-06-01'))
			prod_user_graph[p_id].append((u_id, rating, -1, '2012-06-01'))

	# create evasion ground truth
	user_ground_truth, review_ground_truth = create_evasion_ground_truth(user_product_graph, evasive_spams)


	return [user_priors, review_priors, prod_priors], user_product_graph, prod_user_graph, user_ground_truth, review_ground_truth, new_added_reviews


def add_unmixed_reviews(upg, pug, new_edges, spammer_ids, target_ids, attack_para):

	#### Promotion or demotion setting
	rating = 5
	####
	user_product_graph = cp.deepcopy(upg)
	prod_user_graph = cp.deepcopy(pug)
	new_user_graph = {}
	new_product_graph = {}
	single_mapping = {}

	feature_suspicious_filename = 'feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

	# read the graph and node priors
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()


	if len(new_edges) != 0 and len(spammer_ids) == attack_para[0]:
		# we are under elite attack setting, so we need to load the new added edges and update the features

		# load evasive edges to a new graph
		for added_edge in new_edges:
			added_account = added_edge[0]
			target = added_edge[1]
			origin_date = user_product_graph[added_account][0][3]
			new_date = randomDate(origin_date, 15)
			new_rating = rd.randint(4, 5)
			if added_account not in new_user_graph.keys():
				# a tuple of (product_id, rating, label, posting_date)
				new_user_graph[added_account] = [(target, new_rating, -1, new_date)]
			else:
				new_user_graph[added_account].append((target, new_rating, -1, new_date))
			if target not in new_product_graph.keys():
				# a tuple of (user_id, rating, label, posting_date)
				new_product_graph[target] = [(added_account, new_rating, -1, new_date)]
			else:
				new_product_graph[target].append((added_account, new_rating, -1, new_date))

		# calculate feature_extractors on the complete graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																							  prod_user_graph)
		# update features with the new graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph,
																						   new_user_graph,
																						   prod_user_graph,
																						   new_product_graph,
																						   UserFeatures,
																						   ProdFeatures, ReviewFeatures)
	else:  # we are under the singleton attack setting
		# we need to add the new nodes and edges to the complete graph
		origin_date = '2012-06-01'
		new_date = randomDate(origin_date, 15)
		new_rating = rd.randint(4, 5)
		for iter, target in enumerate(target_ids):
			for user in spammer_ids[iter * 30:iter * 30 + 15]:
				added_account = user
				new_account = str(len(user_product_graph) + len(prod_user_graph))
				single_mapping[added_account] = new_account
				# add the new review to the complete graph
				user_product_graph[new_account] = [(target, new_rating, -1, new_date)]
				prod_user_graph[target].append((new_account, new_rating, -1, new_date))

		# calculate features on the complete graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																							  prod_user_graph)

	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

	user_priors = new_upriors
	review_priors = cp.deepcopy(new_rpriors)
	prod_priors = new_ppriors

	# create ground truth under elite attack
	evasive_spams = {}
	if len(new_edges) == attack_para[1]:
		for added_edge in new_edges:
			added_account = added_edge[0]
			target = added_edge[1]
			if target not in evasive_spams.keys():
				evasive_spams[target] = [(added_account, rating, -1, '2012-06-01')]
			else:
				evasive_spams[target].append((added_account, rating, -1, '2012-06-01'))

	# if we are under the singleton attack setting, we modify their original posts as new spam review
	if len(new_edges) == 0:
		for iter, target in enumerate(target_ids):
			for user in spammer_ids[iter * 30:iter * 30 + 15]:
				if user in single_mapping.keys():
					user = single_mapping[user]
				if target not in evasive_spams.keys():
					evasive_spams[target] = [(user, rating, -1, '2012-06-01')]
				else:
					evasive_spams[target].append((user, rating, -1, '2012-06-01'))

	# add new edge into graph
	# add new edges into the original graph
	for e in new_edges:
		u_id = str(e[0])
		p_id = str(e[1])
		user_product_graph[u_id].append((p_id, rating, -1, '2012-06-01'))
		prod_user_graph[p_id].append((u_id, rating, -1, '2012-06-01'))

	user_ground_truth, review_ground_truth = create_evasion_ground_truth(user_product_graph, evasive_spams)

	return [user_priors, review_priors, prod_priors, new_upriors, new_rpriors, new_ppriors], user_product_graph, prod_user_graph, user_ground_truth, review_ground_truth


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