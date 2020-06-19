import sys
sys.path.insert(0, sys.path[0] + '/..')
from Utils.helper import *
from Detector.SVD import *
import pickle as pkl


def runSVD(new_priors, user_product_graph, percent):
    user_priors = new_priors[0]
    review_priors = new_priors[1]
    prod_priors = new_priors[2]
    model = SVD(user_product_graph, user_priors, prod_priors)
    svd_output = model.run(percent)
    result = model.evaluate_SVD(svd_output, user_product_graph)
    index = list(map(str, map(int, result[0])))
    result_dict = dict(zip(index, result[1]))
    return result_dict


if __name__ == '__main__':
    dataset_name = 'YelpChi'
    prefix = '../Yelp_Dataset/' + dataset_name + '/'
    metadata_filename = prefix + 'metadata.gz'

    # read the graph and node priors
    user_product_graph, product_user_graph = read_graph_data(metadata_filename)
    user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

    # load priors
    with open(prefix + 'priors.pkl', 'rb') as f:
        priors = pkl.load(f)

    percent = 0.9
    userBelief = runSVD(priors, user_product_graph, percent)
    review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
    print('review AUC = {}'.format(review_AUC))
    print('review AP  = {}'.format(review_AP))


