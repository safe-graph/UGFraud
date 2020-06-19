import sys
sys.path.insert(0, sys.path[0] + '/..')
from Detector.gang import *
import pickle as pkl


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
	prefix = '../Yelp_Dataset/' + dataset_name + '/'
	metadata_filename = prefix + 'metadata.gz'

	# load graph and label
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# load priors
	with open(prefix + 'priors.pkl', 'rb') as f:
		priors = pkl.load(f)

	# run GANG model
	model, priors = runGANG(priors, user_product_graph, product_user_graph, user_ground_truth)
	userBelief, _, reviewBelief = model.classify()
	reviewBelief = scale_value(reviewBelief)

	# input parameters: num_iters, stop_threshold

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
