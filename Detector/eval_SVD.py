from Utils.helper import *
from Detector.SVD import *
import sys
import os
sys.path.insert(0, os.path.abspath('../'))


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

    review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
    user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
    product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']
    feature_config = load_feature_config('../Utils/feature_configuration.txt')
    feature_extractor = FeatureExtractor()
    UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
                                                                                          product_user_graph)
    upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
    ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
    rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)
    priors = [upriors, rpriors, ppriors]

    percent = 0.9
    userBelief = runSVD(priors, user_product_graph, percent)
    review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
    print('review AUC = {}'.format(review_AUC))
    print('review AP  = {}'.format(review_AP))


