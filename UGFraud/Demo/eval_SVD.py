from UGFraud.Detector.SVD import *
import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

if __name__ == '__main__':
	# data source
	file_name = 'Yelp_graph_data.json'
	G = load_graph(file_name)
	user_ground_truth = node_attr_filter(G, 'types', 'user', 'label')

	percent = 0.9
	model = SVD(G)
	svd_output = model.run(percent)
	result = model.evaluate_SVD(svd_output, G)
	index = list(map(str, map(int, result[0])))
	userBelief = dict(zip(index, result[1]))
	review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))