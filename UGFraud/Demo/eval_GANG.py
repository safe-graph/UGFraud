from UGFraud.Detector.GANG import *
import sys
import os
sys.path.insert(0, os.path.abspath('../../'))


if __name__ == '__main__':
	# data source
	file_name = 'Yelp_graph_data.json'
	G = load_graph(file_name)
	user_ground_truth = node_attr_filter(G, 'types', 'user', 'label')
	review_ground_truth = edge_attr_filter(G, 'types', 'review', 'label')

	# add semi-supervised user information / threshold
	sup_per = 0.1

	# run GANG model
	model = GANG(G, user_ground_truth, sup_per, nor_flg=True, sup_flg=False)

	# run Linearized Belief Propagation on product-user matrix with 1000 iterations
	iteration = 1000
	model.pu_lbp(iteration)
	userBelief, _, reviewBelief = model.classify()
	reviewBelief = scale_value(reviewBelief)

	# evaluation
	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
