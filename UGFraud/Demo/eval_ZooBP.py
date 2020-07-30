from UGFraud.Utils.helper import *
from UGFraud.Detector.ZooBP import *
import sys
import os
sys.path.insert(0, os.path.abspath('../../'))


if __name__ == '__main__':
    # data source
    file_name = 'Yelp_graph_data.json'
    G = load_graph(file_name)
    user_ground_truth = node_attr_filter(G, 'types', 'user', 'label')

    ep = 0.01
    #  H: compatibility matrix
    H = np.array([[0.5, -0.5], [-0.5, 0.5]])

    model = ZooBP(G, ep, H)
    userBelief, _ = model.run()  # result = (user_beliefs, prod_beliefs)

    review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
    print('review AUC = {}'.format(review_AUC))
    print('review AP  = {}'.format(review_AP))