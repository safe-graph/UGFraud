from Utils.helper import *
from Utils.yelpFeatureExtraction import *
import pickle as pkl

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

	# normalize the review prior as the review suspicious belief
	rpriors = priors[1]
	reviewBelief = scale_value(rpriors)

	review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
	print('review AUC = {}'.format(review_AUC))
	print('review AP  = {}'.format(review_AP))