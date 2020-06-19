import sys
sys.path.insert(0, sys.path[0] + '/..')
from Utils.helper import *
from Detector.ZooBP import *
import pickle as pkl



def runZooBP(priors, adj_list, ep):
    u_priors = priors[0]
    p_priors = priors[2]

    u_tag, user_priors = zip(*u_priors.items())
    user_priors = np.array(user_priors)
    p_tag, prod_priors = zip(*p_priors.items())
    prod_priors = np.array(prod_priors)

    model = ZooBP(adj_list, user_priors, prod_priors, ep)
    user_result, _ = model.run()  # result = (user_beliefs, prod_beliefs)
    userBelief = dict(zip(u_tag, user_result[:, 0]))
    return userBelief


if __name__ == '__main__':
    dataset_name = 'YelpChi'
    prefix = '../Yelp_Dataset/' + dataset_name + '/'
    metadata_filename = prefix + 'metadata.gz'

    # read the graph and node priors
    user_product_graph, product_user_graph, adj_list = read_graph_data(metadata_filename, adj=True)
    user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

    # load graph and label
    user_product_graph, product_user_graph = read_graph_data(metadata_filename)
    user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

    # load priors
    with open(prefix + 'priors.pkl', 'rb') as f:
        priors = pkl.load(f)

    # interaction strength
    ep = 0.01
    userBelief = runZooBP(priors, adj_list, ep)
    review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
    print('review AUC = {}'.format(review_AUC))
    print('review AP  = {}'.format(review_AP))