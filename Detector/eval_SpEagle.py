
from Utils.iohelper import *
from Utils.eval_helper import *
from Detector.SpEagle import *
from Utils.yelpFeatureExtraction import *


def runSpEagle(new_priors, user_product_graph):
	'''
	Run SpEagle with the injected spams
	'''

	numerical_eps = 1e-5
	user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
	eps = 0.1
	review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))

	potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
				  'r_p': review_product_potential, 'p_r': review_product_potential}

	model = SpEagle(user_product_graph, new_priors, potentials, message=None, max_iters=4)

	# new runbp func
	model.schedule(schedule_type='bfs')

	iter = 0
	while iter < 3:

		# set up additional number of iterations
		if iter == 0:
			num_bp_iters = 2
		else:
			num_bp_iters = 1

		message_diff = model.run_bp(start_iter=iter, max_iters=num_bp_iters)

		# print(message_diff)
		
		iter += num_bp_iters

		if message_diff < 1e-3:
			break

	return model


if __name__ == '__main__':

	# dataset source
	dataset_name = 'YelpChi'
	prefix = 'Yelp_Dataset/' + dataset_name + '/'
	metadata_filename = prefix + 'metadata.gz'

	user_product_graph, product_user_graph = read_graph_data(metadata_filename)
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# feature and prior calculation
	feature_suspicious_filename = 'Utils/feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																						  product_user_graph)
	upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)
	priors = [upriors, rpriors, ppriors]

	model = runSpEagle(priors, user_product_graph)
	userBelief, reviewBelief, _ = model.classify()

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
