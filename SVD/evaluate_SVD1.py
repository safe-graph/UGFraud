import pickle
from math import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.stats import gaussian_kde
import sys
sys.path.insert(0, '../../Utils')
from iohelper import *
sys.path.insert(0, '../../SpEagle/featureExtractorPy')
from featureExtraction import *

from SVD import *
#from updated_featureExtraction import *


sys.path.insert(0, '../Utils')
from eval_helper import *





dataset_name = 'YelpChi'
prefix = '../../Yelp_Dataset/' + dataset_name + '/'
prefix1 = '../../Attack/'
threshold = 0.5
metadata_filename = prefix + 'metadata.gz'
user_prior_filename = prefix + 'UserPriorsFromMat.pickle'
prod_prior_filename = prefix + 'ProdPriorsFromMat.pickle'
review_prior_filename = prefix + 'ReviewPriorsFromMat.pickle'

evasion_filename = prefix1 + 'NewYelp/' + 'Random.pickle'
#camou_filename = prefix + 'IncBP/' + 'new_spammer_new_business_no_False.pickle'

# read the graph and node priors
user_product_graph, prod_user_graph = read_graph_data(metadata_filename)


feature_suspicious_filename = 'feature_configuration.txt'
review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

# read the graph and node priors
feature_config = load_feature_config('../../SpEagle/', feature_suspicious_filename)
numerical_eps = 1e-5
user_review_potential = np.log(np.array([[1-numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
eps = 0.1
review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))
	
potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
			'r_p': review_product_potential, 'p_r': review_product_potential}
feature_extractor = FeatureExtractor()
text_features = []
new_user_graph = {}
new_product_graph = {}
	
	
	
pos = set()
node_degree = {}
for u_id, reviews in user_product_graph.items():
    for v in reviews:
        if v[2] == -1:
            pos.add(u_id)
            break
## add edges from evasions and camouflage

with open(evasion_filename, 'rb') as f:
    evasions = pickle.load(f)
    spammer_ids = evasions[0]   
    target_ids = evasions[1]
    new_edges_evasions = evasions[2]

#with open(camou_filename, 'rb') as f:
#    new_edges_camou = pickle.load(f)

#merge new edges from evasions and camouflage
#new_edges = new_edges_evasions + new_edges_camou
new_edges = new_edges_evasions
    

if len(added_edges) != 0 and len(spammers) == 550:
    # we are under mix attack setting, so we need to load the new added edges and update the features

    # load evasive edges to a new graph
    for added_edge in added_edges:
        added_account = added_edge[0]
        target = added_edge[1]
        origin_date = user_product_graph[added_account][0][3]
        new_date = origin_date # randomDate(origin_date)
        new_rating = 5 # rd.randint(4, 5)
        if added_account not in new_user_graph.keys():
            # a tuple of (product_id, rating, label, posting_date)
            new_user_graph[added_account] = [(target, new_rating, -1, new_date)]
        else:
            new_user_graph[added_account].append((target, new_rating, -1, new_date))
        if target not in new_product_graph.keys():
            # a tuple of (user_id, rating, label, posting_date)
            new_product_graph[target] = [(added_account, new_rating, -1, new_date)]
        else:
            new_product_graph[target].append((added_account, new_rating, -1, new_date))
    # add the singleton reviews to complete graph
    origin_date = '2012-06-01'
    new_date = origin_date  # randomDate(origin_date)
    new_rating = 5  # rd.randint(4, 5)
    for iter, target in enumerate(targets):
        for user in spammers[100+iter*15:100+iter*15+15]:
            added_account = user
            user_product_graph[added_account] = [(target, new_rating, -1, new_date)]
            product_user_graph[target].append((added_account, new_rating, -1, new_date))
    # calculate feature_extractorres on the complete graph
    UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
                                                                                          product_user_graph)
    # update features with the new graph
    UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph,
                                                                                           new_user_graph,
                                                                                           product_user_graph,
                                                                                           new_product_graph,
                                                                                           UserFeatures,
                                                                                           ProdFeatures, ReviewFeatures)
else:  # we are under the singleton attack setting
# we need to add the new nodes and edges to the complete graph
    origin_date = '2012-06-01'
    new_date = origin_date  # randomDate(origin_date)
    new_rating = 5  # rd.randint(4, 5)
    for iter, target in enumerate(target_ids):
        for user in spammer_ids[iter*30:iter*30+30]:
            added_account = user
            # add the new review to the complete graph
            user_product_graph[added_account] = [(target, new_rating, -1, new_date)]
            prod_user_graph[target].append((added_account, new_rating, -1, new_date))


    # calculate features on the complete graph
    UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,prod_user_graph)


new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

user_priors = new_upriors
review_priors = new_rpriors
prod_priors = new_ppriors

# create ground truth under mix attack
evasive_spams = {}
if len(added_edges) == 450:
    for added_edge in added_edges:
        added_account = added_edge[0]
        target = added_edge[1]
        if target not in evasive_spams.keys():
            evasive_spams[target] = [(added_account, 5, -1, '2012-06-01')]
        else:
            evasive_spams[target].append((added_account, 5, -1, '2012-06-01'))
    for iter, target in enumerate(targets):
        for user in spammers[100+iter*15:100+iter*15+15]:
            if target not in evasive_spams.keys():
                evasive_spams[target] = [(user, 5, -1, '2012-06-01')]
            else:
                evasive_spams[target].append((user, 5, -1, '2012-06-01'))

# if we are under the singleton attack setting, we modify their original posts as new spam review
if len(new_edges) == 0:
    for iter, target in enumerate(target_ids):
        for user in spammer_ids[iter * 30:iter * 30 + 30]:
            if target not in evasive_spams.keys():
                evasive_spams[target] = [(user, 5, -1, '2012-06-01')]
            else:
                evasive_spams[target].append((user, 5, -1, '2012-06-01'))

user_ground_truth, review_ground_truth = create_evasion_ground_truth(user_product_graph, evasive_spams)

# add new edge into graph
# add new edges into the original graph
for e in new_edges:
    u_id = str(e[0])
    p_id = str(e[1])
    user_product_graph[u_id].append((p_id, 5, 1, '2012-06-01'))
    prod_user_graph[p_id].append((u_id, 5, 1, '2012-06-01')) 


print('run and evaluate SVD')


percent = 0.9 
    
#run SVD on user-product graph
model = SVD(user_product_graph, user_priors, prod_priors)
svd_output = model.run(percent)

#     print(len(svd_output[1,:]))
#     print(svd_output)

#evaluate the SVD based detection with SVM
#result contains the [userid, pred_probas], prediction is the binary pred result, y_ture is the true label vector
result, predictions, y_true = model.evaluate_SVD(svd_output, user_product_graph, user_priors, prod_priors, spammer_ids, percent)
print(y_true)
user_AUC = auc(y_true, predictions)
print(user_AUC)

