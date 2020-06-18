from Utils.helper import *
from Detector.greedy import *
import pickle as pkl


def listToSparseMatrix(edgesSource, edgesDest):
	m = max(edgesSource) + 1
	n = max(edgesDest) + 1
	M = sparse.coo_matrix(([1] * len(edgesSource), (edgesSource, edgesDest)), shape=(m, n))
	M1 = M > 0
	return M1.astype('int')


def runFraudar(new_priors, user_product_graph):

	new_upriors, new_rpriors, new_ppriors = new_priors

	# print('Start detection on the new graph with Fraudar')
	user_to_product = {}
	prod_to_user = {}
	for u_id, reviews in user_product_graph.items():
		if u_id not in user_to_product:
			user_to_product[u_id] = []
		for t in reviews:
			p_id = t[0]
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
	for u_id, reviews in user_product_graph.items():
		for v in reviews:
			p_id = v[0]
			edgesSource.append(u_id2idx[u_id])
			edgesDest.append(p_id2idx[p_id])
	M = listToSparseMatrix(edgesSource, edgesDest)
	# print("finished reading data ")

	res = detect_blocks(M, logWeightedAveDegree)

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
			user_density[u] = 0.000001

	user_prob = {}
	review_prob = {}

	for review in new_rpriors.keys():
		review_prob.update({review: 0.000001})
		user_prob.update({review[0]: 0.000001})
	for user in detected_users.keys():
		user_prob.update({user: user_density[user]})
		for review in user_product_graph[user]:
			review_prob.update({(user, review[0]): user_density[user]})

	return user_prob, review_prob


if __name__ == '__main__':

	dataset_name = 'YelpChi'
	prefix = '../Yelp_Dataset/' + dataset_name + '/'
	metadata_filename = prefix + 'metadata.gz'

	# load graph
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# load priors
	with open(prefix + 'priors.pkl', 'rb') as f:
		priors = pkl.load(f)

	# run Fraudar on the reviews
	userBelief, reviewBelief = runFraudar(priors, user_product_graph)
	reviewBelief = scale_value(reviewBelief)

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))


