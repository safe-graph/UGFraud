
from math import *
import numpy as np
import pickle as pkl
import sys
sys.path.insert(0, sys.path[0] + '/..')

from Utils.iohelper import *
from Utils.yelpFeatureExtraction import *
from Utils.eval_helper import *
from Detector.fBox import *


def runfBox(new_priors, user_product_graph):

	user_priors = new_priors[0]
	review_priors = new_priors[1]
	prod_priors = new_priors[2]

	# print('Start detection on the new graph with fBOX')

	# run fBox
	model = fBox(user_product_graph)
	num_detected_users = []

	################# important parameters
	t = 20  # taus = [0.5, 1, 5, 10, 25, 50, 99]
	k = 50  # k = range(10, 51, 10)
	################# important parameters

	detected_users_by_degree, detected_products_by_degree = model.run(t, k)
	detected_users = set()
	for d, user_list in detected_users_by_degree.items():
		detected_users.update([u for u in user_list])

	num_detected_users.append(len(detected_users))

	detected_products = set()
	for d, prod_list in detected_products_by_degree.items():
		detected_products.update([p for p in prod_list])

	# osrm, isrm = model.get_srms()
	result_uid = []
	user_prob = {}  # result_prob means user_prob
	review_prob = {}
	for u, v in user_priors.items():
		result_uid.append(u)
		if u in detected_users:
			user_prob.update({u: user_priors.get(u)})
		else:
			user_prob.update({u: 0.0000001})

	for user_id, reviews in user_product_graph.items():
		for r in reviews:
			prod_id = r[0]

			if user_id in detected_users:
				review_prob[(user_id, prod_id)] = review_priors.get((user_id, prod_id))
			else:
				review_prob[(user_id, prod_id)] = 0

	return user_prob, review_prob


if __name__ == '__main__':

	dataset_name = 'YelpChi'
	prefix = '../Yelp_Dataset/' + dataset_name + '/'
	metadata_filename = prefix + 'metadata.gz'

	# load graph and label
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# load priors
	with open(prefix + 'priors.pkl', 'rb') as f:
		priors = pkl.load(f)

	userBelief, reviewBelief = runfBox(priors, user_product_graph)
	reviewBelief = scale_value(reviewBelief)

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
