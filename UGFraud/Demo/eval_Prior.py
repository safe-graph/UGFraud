from Utils.helper import *

if __name__ == '__main__':
	# data source
	file_name = 'Yelp_graph_data.json'
	G = load_graph(file_name)
	review_ground_truth = edge_attr_filter(G, 'types', 'review', 'label')

	# normalize the review prior as the review suspicious belief
	rpriors = edge_attr_filter(G, 'types', 'review', 'prior')
	reviewBelief = scale_value(rpriors)

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))
