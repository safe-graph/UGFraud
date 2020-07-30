from UGFraud.Detector.SpEagle import *

if __name__ == '__main__':
	# data source
	file_name = 'Yelp_graph_data.json'
	G = load_graph(file_name)
	review_ground_truth = edge_attr_filter(G, 'types', 'review', 'label')

	# input parameters: numerical_eps, eps, num_iters, stop_threshold
	numerical_eps = 1e-5
	eps = 0.1
	user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
	review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))
	potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
	              'r_p': review_product_potential, 'p_r': review_product_potential}
	max_iters = 4
	stop_threshold = 1e-3

	model = SpEagle(G, potentials, message=None, max_iters=4)

	# new runbp func
	model.schedule(schedule_type='bfs')

	iter = 0
	num_bp_iters = 2
	model.run_bp(start_iter=iter, max_iters=num_bp_iters, tol=stop_threshold)

	userBelief, reviewBelief, _ = model.classify()

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
