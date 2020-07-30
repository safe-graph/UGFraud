from UGFraud.Utils.helper import *
from UGFraud.Detector.fBox import *


def runfBox(graph, t, k):
    user_priors = node_attr_filter(graph, 'types', 'user', 'prior')
    review_priors = edge_attr_filter(graph, 'types', 'review', 'prior')

    # run fBox
    model = fBox(graph)
    num_detected_users = []

    detected_users_by_degree, detected_products_by_degree = model.run(t, k)
    detected_users = set()
    for d, user_list in detected_users_by_degree.items():
        detected_users.update([u for u in user_list])

    num_detected_users.append(len(detected_users))

    detected_products = set()
    for d, prod_list in detected_products_by_degree.items():
        detected_products.update([p for p in prod_list])

    result_uid = []
    user_prob = {}  # result_prob means user_prob
    review_prob = {}
    for u, v in user_priors.items():
        result_uid.append(u)
        if u in detected_users:
            user_prob.update({u: user_priors.get(u)})
        else:
            user_prob.update({u: 1e-7})

    for user_prod in graph.edges:
        if user_prod[0] in detected_users:
            review_prob[(user_prod[0], user_prod[1])] = review_priors.get((user_prod[0], user_prod[1]))
        else:
            review_prob[(user_prod[0], user_prod[1])] = 0

    return user_prob, review_prob


if __name__ == '__main__':
    # data source
    file_name = 'Yelp_graph_data.json'
    G = load_graph(file_name)
    review_ground_truth = edge_attr_filter(G, 'types', 'review', 'label')

    # important parameters
    t = 20  # taus = [0.5, 1, 5, 10, 25, 50, 99]
    k = 50  # k = range(10, 51, 10)

    userBelief, reviewBelief = runfBox(G, t, k)

    reviewBelief = scale_value(reviewBelief)

    review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
    print('review AUC = {}'.format(review_AUC))
    print('review AP  = {}'.format(review_AP))
