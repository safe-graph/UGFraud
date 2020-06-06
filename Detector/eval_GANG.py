
from Utils.iohelper import *
from Utils.eval_helper import *
from Detector.gang import *
from Utils.yelpFeatureExtraction import *


def runGANG(priors, user_product_graph, product_user_graph, user_ground_truth):
	"""
	Run GANG model
	"""

	# need normalized the prior before running GANG
	priors, mean_priors = nor_priors(priors)

	model = GANG(user_product_graph, product_user_graph, user_ground_truth,
				 priors, mean_priors, 0.1, nor_flg=True, sup_flg=False)

	# run Linearized Belief Propagation on product-user matrix with 1000 iterations
	model.pu_lbp(1000)

	return model, [priors[0], priors[1], priors[2]]


if __name__ == '__main__':
	dataset_name = 'YelpChi'
	prefix = 'Yelp_Dataset/' + dataset_name + '/'
	metadata_filename = prefix + 'metadata.gz'

	# load graph and label
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

	# run GANG model
	model, priors = runGANG(priors, user_product_graph, product_user_graph, user_ground_truth)
	userBelief, _, reviewBelief = model.classify()
	reviewBelief = scale_value(reviewBelief)

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
