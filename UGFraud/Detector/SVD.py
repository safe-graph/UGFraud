"""
    'Singular Value Decomposition and Least Squares Solutions'
    The Singular-Value Decomposition, or SVD for short, is a matrix decomposition method for reducing
    a matrix to its constituent parts in order to make certain subsequent matrix calculations simpler.
    Article: https://link.springer.com/content/pdf/10.1007/978-3-662-39778-7_10.pdf
"""

from UGFraud.Utils.helper import *
from sklearn import svm
from sklearn.svm import SVC
from scipy.sparse.linalg import svds
import numpy as np


class SVD:
    def __init__(self, graph):
        """set up the data
        Args:
            graph: a networkx graph
        """
        user_priors = node_attr_filter(graph, 'types', 'user', 'prior')
        prod_priors = node_attr_filter(graph, 'types', 'prod', 'prior')
        num_users = len(user_priors)
        num_products = len(prod_priors)
        self.user_prod_matrix = np.empty(shape=(num_users, num_products))

        # create a dict for user_index in the user list and a dict for prod_index in the product list
        self.user_index = dict()
        self.prod_index = dict()

        i = 0
        for u_id in user_priors.keys():
            self.user_index[u_id] = i
            i = i + 1

        j = 0
        for prod_id in prod_priors.keys():
            self.prod_index[prod_id] = j
            j = j + 1
        #
        for user_id in user_priors.keys():
            for p_id in graph[user_id].keys():
                rating = graph.edges.get((user_id, p_id))['rating']
                row = self.user_index[user_id]
                column = self.prod_index[p_id]
                self.user_prod_matrix[row, column] = rating

    @timer
    def run(self, percent):
        """
        perform SVD and return the user-product matrix in a lower dimensional space
        """
        k = int(max(np.round(min(self.user_prod_matrix.shape) * percent), 1))
        u, s, v = svds(self.user_prod_matrix, k=k)
        return u

    def random_split(self, graph):
        """
            Partition user nodes into training and test set randomly.
            Args:
                user_product_graph: a dictionary, with key = user_id, value = (p_id, rating, label, time)
            Return:
                training_user_id: a set of user id to appear in model training
        """
        pos = set()
        node_degree = {}
        user_dict = node_attr_filter(graph, 'types', 'user', 'types')
        for u_id in user_dict.keys():
            for p_id in graph[u_id].keys():
                if graph.edges.get((u_id, p_id))['label'] == 0:
                    pos.add(u_id)
                    break
            node_degree[u_id] = len(graph[u_id])

        neg = set(list(user_dict.keys())) - pos

        # random sample positive users
        training_pos = set(np.random.choice(list(pos), int(0.5 * len(pos))).ravel())
        training_neg = set(np.random.choice(list(neg), int(0.5 * len(neg))).ravel())

        test_pos = pos - training_pos
        test_neg = neg - training_neg

        print("number of positive %d" % len(pos))
        print("number of negative %d" % len(neg))
        print("number of all users %d" % len(user_dict))

        return training_pos, training_neg, test_pos, test_neg

    def classify(self, training_data_svm, training_labels_svm, testing_data_svm, testing_labels_svm):
        clf = svm.SVC(probability=True)
        clf.fit(training_data_svm, training_labels_svm)
        SVC(C=100, tol=0.00001)
        predictions = clf.predict_proba(testing_data_svm)
        return predictions

    def classify_binary(self, training_data_svm, training_labels_svm, testing_data_svm, testing_labels_svm):
        clf = svm.SVC()
        clf.fit(training_data_svm, training_labels_svm)
        SVC(C=100, tol=0.00001)
        predictions = clf.predict(testing_data_svm)
        return predictions

    def evaluate_SVD(self, svd_output, graph):
        # random_split
        training_pos, training_neg, test_pos, test_neg = self.random_split(graph)
        training_labels = {i: +1 for i in training_pos}
        training_labels.update({i: -1 for i in training_neg})

        test_labels = {i: +1 for i in test_pos}
        test_labels.update({i: -1 for i in test_neg})

        training_data_svm = np.empty(shape=(len(training_labels), len(svd_output[1, :])))
        training_labels_svm = np.empty(shape=(len(training_labels)))
        # build training data and labels for svm
        i = 0
        find_training_uid = dict()
        for k, v in training_labels.items():
            u_index = self.user_index[k]
            training_data_svm[i, :] = svd_output[u_index, :]
            training_labels_svm[i] = v
            find_training_uid[i] = k
            i = i + 1
        # build testing data and labels for svm
        testing_data_svm = np.empty(shape=(len(test_labels), len(svd_output[1, :])))
        testing_labels_svm = np.empty(shape=(len(test_labels)))
        j = 0
        find_testing_uid = np.empty(shape=(len(test_labels)))
        for k, v in test_labels.items():
            u_index = self.user_index[k]
            testing_data_svm[j, :] = svd_output[u_index, :]
            testing_labels_svm[j] = v
            find_testing_uid[j] = k
            j = j + 1

        probas_pred = self.classify(training_data_svm, training_labels_svm, testing_data_svm, testing_labels_svm)
        result = [find_testing_uid, probas_pred[:, 0]]
        return result