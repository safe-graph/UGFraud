"""
	Implement SVD on a User-Product graph.
"""
from sklearn import svm
from sklearn.svm import SVC
from scipy.sparse.linalg import svds

import numpy as np


class SVD:
    def __init__(self, user_product_graph, user_priors, prod_priors):
        """set up the data
        Args:
            user_product_graph: a dictionary, with key = user_id, value = (p_id, rating, label, time)
            priors: a tuple of prior matrices (user_priors, product_priors, review_prior) in probability scale ([0,1])
            potentials: a dictionary (key = edge_type, value=np.ndarray)
        """
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
        for user_id, reviews in user_product_graph.items():

            for r in reviews:
                prod_id = r[0]
                rating = r[1]
                row = self.user_index[user_id]
                column = self.prod_index[prod_id]
                self.user_prod_matrix[row, column] = rating

    def run(self, percent):
        """
        perform SVD and return the user-product matrix in a lower dimensional space
        """
        k = int(max(np.round(min(self.user_prod_matrix.shape) * percent), 1))
        u, s, v = svds(self.user_prod_matrix, k=k)
        print(s)
        return u

    def random_split(self, user_product_graph):
        """
            Partition user nodes into training and test set randomly.
            Args:
                user_product_graph: a dictionary, with key = user_id, value = (p_id, rating, label, time)
            Return:
                training_user_id: a set of user id to appear in model training
        """
        pos = set()
        node_degree = {}
        for u_id, reviews in user_product_graph.items():
            for v in reviews:
                if v[2] == -1:
                    pos.add(u_id)
                    break
            node_degree[u_id] = len(reviews)

        neg = set(list(user_product_graph.keys())) - pos

        # random sample positive users
        training_pos = set(np.random.choice(list(pos), int(0.5 * len(pos))).ravel())
        training_neg = set(np.random.choice(list(neg), int(0.5 * len(neg))).ravel())

        test_pos = pos - training_pos
        test_neg = neg - training_neg

        print("number of positive %d" % len(pos))
        print("number of negative %d" % len(neg))
        print("number of all users %d" % len(user_product_graph))

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

    def evaluate_SVD(self, svd_output, user_prod_graph):
        # random_split
        training_pos, training_neg, test_pos, test_neg = self.random_split(user_prod_graph)
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
        result = [find_testing_uid, probas_pred[:,0]]
        return result



