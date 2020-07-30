"""
	'FRAUDAR: Bounding Graph Fraud in the Face of camouflage'
	Spot fraudsters in the presence of camouflage or hijacked accounts. An algorithm that is camouflage-resistant,
	provides upper bounds on the effectiveness of fraudsters, and the algorithm is effective in real-world data.
	Article: https://bhooi.github.io/papers/fraudar_kdd16.pdf
"""

from UGFraud.Utils.helper import *
from UGFraud.Detector.greedy import *
import copy as cp
import sys
import os
sys.path.insert(0, os.path.abspath('../../'))


def listToSparseMatrix(edgesSource, edgesDest):
	m = max(edgesSource) + 1
	n = max(edgesDest) + 1
	M = sparse.coo_matrix(([1] * len(edgesSource), (edgesSource, edgesDest)), shape=(m, n))
	M1 = M > 0
	return M1.astype('int')


@timer
def runFraudar(graph, multiple=0):
	new_upriors = node_attr_filter(graph, 'types', 'user', 'prior')
	new_rpriors = edge_attr_filter(graph, 'types', 'review', 'prior')
	# print('Start detection on the new graph with Fraudar')
	user_to_product = {}
	prod_to_user = {}
	u_id_dict = node_attr_filter(graph, 'types', 'user', 'types')
	for u_id in u_id_dict.keys():
		if u_id not in user_to_product:
			user_to_product[u_id] = []
		for p_id in graph[u_id].keys():
			if p_id not in prod_to_user:
				prod_to_user[p_id] = []
				user_to_product[u_id].append(p_id)
				prod_to_user[p_id].append(u_id)
	u_id2idx = {}
	p_id2idx = {}
	idx2u_id = {}
	idx2p_id = {}
	i = 0
	for u_id in user_to_product.keys():
		u_id2idx[u_id] = i
		idx2u_id[i] = u_id
		i += 1

	i = 0
	for p_id in prod_to_user.keys():
		p_id2idx[p_id] = i
		idx2p_id[i] = p_id
		i += 1

	edgesSource = []
	edgesDest = []
	for u_id in u_id_dict.keys():
		for p_id in graph[u_id].keys():
			edgesSource.append(u_id2idx[u_id])
			edgesDest.append(p_id2idx[p_id])
	M = listToSparseMatrix(edgesSource, edgesDest)
	# print("finished reading data ")

	if multiple == 0:
		# detect all dense blocks 
		res = detect_blocks(M, logWeightedAveDegree)
	else:
		# detect the top #multiple dense blocks
		res = detectMultiple(M, logWeightedAveDegree, multiple)

	detected_users = {}
	weight_dict = {}
	for lwRes in res:
		detected_u_idx = lwRes[0][0]
		detected_p_idx = lwRes[0][1]
		weight = lwRes[1]
		weight_dict[weight] = weight
		for i in detected_u_idx:
			uid_tmp = idx2u_id[i]
			if uid_tmp not in detected_users.keys():
				detected_users[uid_tmp] = weight

	max_den = res[0][1]
	min_den = res[-1][1]
	den_interval = max_den - min_den

	ranked_rpriors = [(review, new_rpriors[review]) for review in new_rpriors.keys()]
	ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]
	aux_rpriors = cp.deepcopy(new_rpriors)
	for i, p in aux_rpriors.items():
		new_rpriors[i] = (p - r_min) / (r_max - r_min)

	user_density = {}
	for u in new_upriors.keys():
		if u in detected_users.keys():
			user_density[u] = (detected_users[u] - min_den) / den_interval
		else:
			user_density[u] = 1e-6

	user_prob = {}
	review_prob = {}
	for review in new_rpriors.keys():
		review_prob.update({review: 1e-6})
		user_prob.update({review[0]: 1e-6})
	print(len(detected_users))
	print(detected_users['302'])

	for user in detected_users.keys():
		user_prob.update({user: user_density[user]})
		for prod in graph[user].keys():
			review_prob.update({(user, prod): user_density[user]})

	return user_prob, review_prob


if __name__ == '__main__':
	# data source
	file_name = 'Yelp_graph_data.json'
	G = load_graph(file_name)
	review_ground_truth = edge_attr_filter(G, 'types', 'review', 'label')

	# run Fraudar on the reviews
	userBelief, reviewBelief = runFraudar(G, multiple=0)
	reviewBelief = scale_value(reviewBelief)

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))


