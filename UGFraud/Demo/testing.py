from UGFraud.Demo.eval_fBox import *
from UGFraud.Demo.eval_Fraudar import *
from UGFraud.Demo.eval_GANG import *
from UGFraud.Demo.eval_SpEagle import *
from UGFraud.Demo.eval_SVD import *
from UGFraud.Demo.eval_ZooBP import *
from UGFraud.Demo.data_to_network_graph import *
import sys
import os

sys.path.insert(0, os.path.abspath('../../'))

# data source
file_name = 'Yelp_graph_data.json'
try:
    G = load_graph(file_name)
except FileNotFoundError:
    data_to_network_graph()
    G = load_graph(file_name)
user_ground_truth = node_attr_filter(G, 'types', 'user', 'label')
review_ground_truth = edge_attr_filter(G, 'types', 'review', 'label')

"""
    testing fBox
"""
print("*" * 80)
print("Testing fBox")
t = 20  # taus = [0.5, 1, 5, 10, 25, 50, 99]
k = 50  # k = range(10, 51, 10)
serBelief, reviewBelief = runfBox(G, t, k)
reviewBelief = scale_value(reviewBelief)

review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
print('review AUC = {}'.format(review_AUC))
print('review AP  = {}'.format(review_AP))

"""
    testing Fraudar
"""
print("*" * 80)
print("Testing Fraudar")
userBelief, reviewBelief = runFraudar(G, multiple=0)
reviewBelief = scale_value(reviewBelief)

review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
print('review AUC = {}'.format(review_AUC))
print('review AP  = {}'.format(review_AP))

"""
    testing GANG
"""
print("*" * 80)
print("Testing GANG")
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

"""
    testing Prior
"""
print("*" * 80)
print("Testing Prior")
# normalize the review prior as the review suspicious belief
rpriors = edge_attr_filter(G, 'types', 'review', 'prior')
reviewBelief = scale_value(rpriors)

review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
print('review AUC = {}'.format(review_AUC))
print('review AP  = {}'.format(review_AP))

"""
    testing SpEagle
"""
print("*" * 80)
print("Testing SpEagle")
# input parameters: numerical_eps, eps, num_iters, stop_threshold
numerical_eps = 1e-5
eps = 0.1
user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))
potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
              'r_p': review_product_potential, 'p_r': review_product_potential}
max_iters = 4
stop_threshold = 1e-3

model = SpEagle(G, potentials, message=None, max_iters=4)

# new runbp func
model.schedule(schedule_type='bfs')

iter = 0
num_bp_iters = 2
model.run_bp(start_iter=iter, max_iters=num_bp_iters, tol=stop_threshold)

userBelief, reviewBelief, _ = model.classify()

review_AUC, review_AP = evaluate(review_ground_truth, reviewBelief)
print('review AUC = {}'.format(review_AUC))
print('review AP  = {}'.format(review_AP))

"""
    testing SVG
"""
print("*" * 80)
print("Testing SVD")
percent = 0.9
model = SVD(G)
svd_output = model.run(percent)
result = model.evaluate_SVD(svd_output, G)
index = list(map(str, map(int, result[0])))
userBelief = dict(zip(index, result[1]))
review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
print('review AUC = {}'.format(review_AUC))
print('review AP  = {}'.format(review_AP))

"""
    testing ZooBP
"""
print("*" * 80)
print("Testing ZooBp")
ep = 0.01
#  H: compatibility matrix
H = np.array([[0.5, -0.5], [-0.5, 0.5]])

model = ZooBP(G, ep, H)
userBelief, _ = model.run()  # result = (user_beliefs, prod_beliefs)

review_AUC, review_AP = evaluate(user_ground_truth, userBelief)
print('review AUC = {}'.format(review_AUC))
print('review AP  = {}'.format(review_AP))
