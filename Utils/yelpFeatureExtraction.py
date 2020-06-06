"""
	Construct account, product and review features given a review dataset.
"""
import math
import pickle
import sys
from copy import deepcopy
from datetime import datetime
import numpy as np

from Utils.iohelper import *


class FeatureExtractor:
    date_time_format_str = '%Y-%m-%d'

    def __init__(self):
        # for the users, keep the normalizer of mnr
        self.user_mnr_normalizer = 0

        # for the products, keep the normalizer of mnr
        self.prod_mnr_normalizer = 0

        # keeping number of reviews and sum of ratings for each product
        self.product_num_ratings = {}
        self.product_sum_ratings = {}

    def MNR(self, data, data_type='user'):
        """
            Normalized maximum number of reviews in a day for a user/product
            Args:
                data is a dictionary with key=u_id or p_id and value = tuples of (neighbor id, rating, label, posting time)
            Return:
                dictionary with key = u_id or p_id and value = MNR
        """
        # maximum number of reviews written in a day for user / product
        feature = {}
        for i, d in data.items():
            # key = posting date; value = number of reviews
            frequency = {}
            for t in d:
                if t[3] not in frequency:
                    frequency[t[3]] = 1
                else:
                    frequency[t[3]] += 1
            feature[i] = max(frequency.values())

        # normalize it
        if data_type == 'user':
            self.user_mnr_normalizer = max(feature.values())
            for k in feature.keys():
                feature[k] /= self.user_mnr_normalizer
        else:
            self.prod_mnr_normalizer = max(feature.values())
            for k in feature.keys():
                feature[k] /= self.prod_mnr_normalizer

        return feature

    def iMNR(self, data, new_data, data_type='user'):
        """ Incremental version of MNR
        """
        feature = {}
        for i, d in new_data.items():
            all_d = deepcopy(d)
            if i in data:
                all_d += data[i]

            frequency = {}
            for t in all_d:
                if t[3] not in frequency:
                    frequency[t[3]] = 1
                else:
                    frequency[t[3]] += 1
            feature[i] = max(frequency.values())

        # normalize it
        if data_type == 'user':
            self.user_mnr_normalizer = max(max(feature.values()), self.user_mnr_normalizer)
            for k in feature.keys():
                feature[k] /= self.user_mnr_normalizer
        else:
            self.prod_mnr_normalizer = max(max(feature.values()), self.prod_mnr_normalizer)
            for k in feature.keys():
                feature[k] /= self.prod_mnr_normalizer

        return feature

    def PR_NR(self, data):
        """
            Ratio of positive and negative reviews of a user or product
            Args:
                data is a dictionary with key=u_id or p_id and value = tuples of (neighbor id, rating, label, posting time)
            Return:
                dictionary with key = u_id or p_id and value = (PR, NR)
        """
        feature = {}

        for i, d in data.items():
            positives = [1 for t in d if t[1] > 3]
            negatives = [1 for t in d if t[1] < 3]
            feature[i] = (float(len(positives)) / len(d), float(len(negatives)) / len(d))
        return feature

    def iPR_NR(self, data, new_data):
        feature = {}

        for i, d in new_data.items():
            all_d = deepcopy(d)
            if i in data:
                all_d = all_d + data[i]
            positives = [1 for t in all_d if t[1] > 3]
            negatives = [1 for t in all_d if t[1] < 3]
            feature[i] = (float(len(positives)) / len(all_d), float(len(negatives)) / len(all_d))

        return feature

    def avgRD_user(self, user_data, product_data):
        """
            Average rating deviation of each user / product.
            For a user i, avgRD(i) = average(r_ij - avg_j | for all r_ij of the user i)
            For a product j, avgRD(j) = average(r_ij - avg_j | for all r_ij of the user i) = 0!?
            Return:
                average rating deviation on users, as defined in the paper
                Detecting product review spammers using rating behaviors, CIKM, 2010
        """
        p_avg = {}
        # find the average rating of each product
        for i, d in product_data.items():
            self.product_num_ratings[i] = len(d)
            self.product_sum_ratings[i] = np.sum(np.array([t[1] for t in d]))
            p_avg[i] = np.mean(np.array([t[1] for t in d]))
        # find average rating deviation of each user
        u_avgRD = {}
        for i, d in user_data.items():
            u_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[t[0]]) for t in d]))

        # if i == '202':
        #    print (u_avgRD[i])
        #    for r in d:
        #        print (p_avg[r[0]])
        return u_avgRD

    def iavgRD_user(self, user_data, new_user_data, product_data, new_product_data):
        """
            Need to ensure those target products in new_user_data are also in new_product_data
        """
        # the users whose avgRD needs update
        user_involved = set()

        # update count and sum of ratings for each product
        for i, d in new_product_data.items():
            # the product is an existing one
            if i in self.product_num_ratings:
                self.product_num_ratings[i] += len(d)
            # or it can be a new product
            else:
                self.product_num_ratings[i] = len(d)
            # go through the new reviews of each product in new_product_data
            for t in d:
                if i in self.product_sum_ratings:
                    self.product_sum_ratings[i] += t[1]
                else:
                    self.product_sum_ratings[i] = t[1]

                user_id = t[0]
                # print ('new user id for product %s:%s' % (i, user_id))
                user_involved.add(user_id)
            # go through the existing reviews of each product in new_product_data
            if i in product_data:
                # print ('existing user id for product %s:' % i)
                for t in product_data[i]:
                    user_id = t[0]
                    #	                print (user_id)
                    user_involved.add(user_id)

        # verify that all new users are in user_involved
        for i, d in new_user_data.items():
            assert i in user_involved
        # find averaged ratings of the products involved
        p_avg = {}
        # for i, d in new_product_data.items():
        #    all_d = deepcopy(d)
        #    if i in product_data:
        #        all_d += product_data[i]

        #    p_avg[i] = np.mean(np.array([t[1] for t in all_d]))

        # find average rating deviation of each user who is involved
        u_avgRD = {}
        for user_id in user_involved:
            # the involved user must be in at least one of new_user_data or user_data
            all_d = []
            if user_id in new_user_data:
                all_d += new_user_data[user_id]
            if user_id in user_data:
                all_d += user_data[user_id]
            # go thru all targets (including new and existing ones) of user i
            for r in all_d:
                product_id = r[0]
                if product_id not in p_avg:
                    p_avg[product_id] = self.product_sum_ratings[product_id] / self.product_num_ratings[product_id]
                # for r in all_d:
                #    print ('user %s: rating %d' % (user_id, r[1]))
                # print ('product %s: avg rating %f' % (r[0], p_avg[r[0]]))
            u_avgRD[user_id] = np.mean(np.array([abs(r[1] - p_avg[r[0]]) for r in all_d]))

        # if i == '202':
        #    print (u_avgRD[i])
        #    for r in all_d:
        #        print (p_avg[r[0]])
        return u_avgRD

    def avgRD_prod(self, product_data):
        """
            Average rating deviation of each user / product.
            For a user i, avgRD(i) = average(r_ij - avg_j | for all r_ij of the product i)
            For a product j, avgRD(j) = average(r_ij - avg_j | for all r_ij of the product i) = 0!?
            Return:
                average rating deviation on products, as defined in the paper
                collective opinion spam detection: bridging review networks and metadata, KDD, 2015
        """
        p_avg = {}
        # find the average rating of each product
        for i, d in product_data.items():
            self.product_num_ratings[i] = len(d)
            self.product_sum_ratings[i] = np.sum(np.array([t[1] for t in d]))
            p_avg[i] = np.mean(np.array([t[1] for t in d]))

        # find average rating deviation of each product
        p_avgRD = {}
        for i, d in product_data.items():
            p_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[i]) for t in d]))

        return p_avgRD

    def iavgRD_prod(self, product_data, new_product_data):

        # update count and sum of ratings for each product
        for i, d in new_product_data.items():
            if i in self.product_num_ratings:
                self.product_num_ratings[i] += len(d)
            else:
                self.product_num_ratings[i] = len(d)
            for t in d:
                if i in self.product_sum_ratings:
                    self.product_sum_ratings[i] += t[1]
                else:
                    self.product_sum_ratings[i] = t[1]

        # find the average rating of each product
        p_avg = {}
        p_avgRD = {}
        for i, d in new_product_data.items():
            all_d = deepcopy(d)
            if i in product_data:
                # this will modify the contents of all_d and also the new_product_data!
                all_d += product_data[i]
            p_avg[i] = np.mean(np.array([t[1] for t in all_d]))
            p_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[i]) for t in all_d]))
        #    if i == '1':
        #        print (len(all_d))
        #        print (p_avg[i])

        # find average rating deviation of each product
        # p_avgRD = {}
        # for i, d in new_product_data.items():
        #    all_d = d
        #    if i == '1':
        #        print (len(all_d))
        #    if i in product_data:
        #        all_d += product_data[i]
        #    if i == '1':
        #        print (len(all_d))
        #    p_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[i]) for t in all_d]))

        return p_avgRD

    def BST(self, user_data):
        """ Burstiness of reviews by users. Spammers are often short term
            members of the site: so BST(i) = 0, if L(i) - F(i) > tau else BST(i) = 1 -
            (L(i) - F(i))/tau, where, L(i) - F(i) are number of days between first and
            last review of i, tau = 28 days

            Args:
                user_data is a dictionary with key=u_id value = tuples of (prod_id, rating, label, posting time)
            Return:
                dictionary with key = u_id and value = BST
        """
        bst = {}
        tau = 28.0  # 28 days
        for i, d in user_data.items():
            post_dates = sorted([datetime.strptime(t[3], self.date_time_format_str) for t in d])
            delta_days = (post_dates[-1] - post_dates[0]).days
            if delta_days > tau:
                bst[i] = 0.0
            else:
                bst[i] = 1.0 - (delta_days / tau)
        return bst

    def iBST(self, user_data, new_user_data):
        bst = {}
        tau = 28.0  # 28 days
        for i, d in new_user_data.items():
            all_d = deepcopy(d)
            if i in user_data:
                all_d += user_data[i]
            post_dates = sorted([datetime.strptime(t[3], self.date_time_format_str) for t in all_d])
            delta_days = (post_dates[-1] - post_dates[0]).days
            if delta_days > tau:
                bst[i] = 0.0
            else:
                bst[i] = 1.0 - (delta_days / tau)
        return bst

    def ERD(self, data):
        """
            Entropy of the rating distribution of each user (product)
        """
        erd = {}
        for i, d in data.items():
            ratings = [t[1] for t in d]
            h, _ = np.histogram(ratings, bins=np.arange(1, 7))
            h = h / h.sum()
            h = h[np.nonzero(h)]
            erd[i] = (- h * np.log2(h)).sum()
        return erd

    def iERD(self, data, new_data):
        erd = {}
        for i, d in new_data.items():
            all_d = deepcopy(d)
            if i in data:
                all_d += data[i]
            ratings = [t[1] for t in all_d]
            h, _ = np.histogram(ratings, bins=np.arange(1, 7))
            h = h / h.sum()
            h = h[np.nonzero(h)]
            erd[i] = (- h * np.log2(h)).sum()
        return erd

    def ETG(self, data):
        """
            Entropy of the gaps between any two consecutive ratings.
        """
        etg = {}
        # [0,1) -> 1
        # [1,3) -> 2
        # ...
        # [17, 33) -> 6
        # anything larger than 33 will be discarded.

        edges = [0, 0.5, 1, 4, 7, 13, 33]
        for i, d in data.items():

            # if there is only one posting time, then entropy = 0
            if len(d) <= 1:
                etg[i] = 0
                continue
            # sort posting dates from the past to the future
            posting_dates = sorted([datetime.strptime(t[3], self.date_time_format_str) for t in d])

            # find the difference in days between two consecutive dates
            delta_days = [(posting_dates[i + 1] - posting_dates[i]).days for i in range(len(posting_dates) - 1)]
            delta_days = [d for d in delta_days if d < 33]

            # bin to the 6 bins, discarding any differences that are greater than 33 days
            h = []
            for delta in delta_days:
                j = 0
                while j < len(edges) and delta > edges[j]:
                    j += 1
                h.append(j)
            _, h = np.unique(h, return_counts=True)
            if h.sum() == 0:
                etg[i] = 0
                continue
            h = h / h.sum()
            h = h[np.nonzero(h)]
            etg[i] = np.sum(- h * np.log2(h))
        return etg

    def iETG(self, data, new_data):
        etg = {}
        edges = [0, 0.5, 1, 4, 7, 13, 33]
        for i, d in new_data.items():
            all_d = deepcopy(d)
            if i in data:
                all_d += data[i]
            # if there is only one posting time, then entropy = 0
            if len(all_d) <= 1:
                etg[i] = 0
                continue
            # sort posting dates from the past to the future
            posting_dates = sorted([datetime.strptime(t[3], self.date_time_format_str) for t in all_d])

            # find the difference in days between two consecutive dates
            delta_days = [(posting_dates[i + 1] - posting_dates[i]).days for i in range(len(posting_dates) - 1)]
            delta_days = [d for d in delta_days if d < 33]

            # bin to the 6 bins, discarding any differences that are greater than 33 days
            h = []
            for delta in delta_days:
                j = 0
                while j < len(edges) and delta > edges[j]:
                    j += 1
                h.append(j)
            _, h = np.unique(h, return_counts=True)
            if h.sum() == 0:
                etg[i] = 0
                continue
            h = h / h.sum()
            h = h[np.nonzero(h)]
            etg[i] = np.sum(- h * np.log2(h))
        return etg

    def RD(self, product_data):
        """Calculate the deviation of the review ratings to the product average.

            Args:
                prod_data:
            Return:
                a dictionary with key = (u_id, p_id), value = deviation of the rating of this review to the average rating of the target product
        """
        rd = {}
        for i, d in product_data.items():
            avg = np.mean(np.array([t[1] for t in d]))
            for t in d:
                rd[(t[0], i)] = abs(t[1] - avg)
        return rd

    def iRD(self, product_data, new_product_data):
        rd = {}
        for i, d in new_product_data.items():
            all_d = deepcopy(d)
            if i in product_data:
                all_d = d + product_data[i]
            avg = np.mean(np.array([t[1] for t in all_d]))
            for t in all_d:
                rd[(t[0], i)] = abs(t[1] - avg)
        return rd

    def EXT(self, product_data):
        """
            Whether a rating is extreme or not
            Args:
                product_data is a dictionary with key=p_id and value = tuples of (u_id, rating, label, posting time)
            Return:
                a dictionary with key = (u_id, p_id) and value = 0 (not extreme) / 1 (extreme)
        """
        ext = {}
        for i, d in product_data.items():
            for t in d:
                if int(t[1]) == 5 or int(t[1]) == 1:
                    ext[(t[0], i)] = 1
                else:
                    ext[(t[0], i)] = 0
        return ext

    def iEXT(self, product_data, new_product_data):
        ext = {}
        for i, d in new_product_data.items():
            all_d = deepcopy(d)
            if i in product_data:
                all_d = d + product_data[i]
            for t in all_d:
                if int(t[1]) == 5 or int(t[1]) == 1:
                    ext[(t[0], i)] = 1
                else:
                    ext[(t[0], i)] = 0
        return ext

    def DEV(self, product_data):
        """
            Deviation of each rating from the average rating of the target product.
            Need to use "recursive minimal entropy partitioning" to find beta_1
            Args:
                product_data is a dictionary with key=p_id and value = tuples of (neighbor id, rating, label, posting time)
            Return:
                a dictionary with key = (u_id, p_id) and value = (RD_ij, RD_ij / 4 > 0.63 ? 1: 0)
                where RD_ij = |r_ij - average rating of product j|
        """
        beta_1 = 0.63
        dev = {}
        # i is a product id
        for i, d in product_data.items():
            # find the average rating of each product
            p_avg_rating = np.mean(np.array([t[1] for t in d]))
            for t in d:
                u_id = t[0]  # user id
                if (abs(p_avg_rating - t[1]) / 4.0 > 0.63):
                    dev[(u_id, i)] = 1  # absolute difference between current rating and product average rating
                else:
                    dev[(u_id, i)] = 0  # absolute difference between current rating and product average rating
        return dev

    def iDEV(self, product_data, new_product_data):
        beta_1 = 0.63
        dev = {}
        # i is a product id
        for i, d in new_product_data.items():
            all_d = deepcopy(d)
            if i in product_data:
                all_d = d + product_data[i]
            # find the average rating of each product
            p_avg_rating = np.mean(np.array([t[1] for t in all_d]))
            for t in all_d:
                u_id = t[0]  # user id
                if (abs(p_avg_rating - t[1]) / 4.0 > 0.63):
                    dev[(u_id, i)] = 1  # absolute difference between current rating and product average rating
                else:
                    dev[(u_id, i)] = 0  # absolute difference between current rating and product average rating
        return dev

    def ETF(self, product_data):
        """
            Binary feature: 0 if ETF_prime <= beta_3, 1 otherwise.
            Need to use "recursive minimal entropy partitioning" to find beta_1
            ETF_prime = 1 - (date of last review of user i on product p from the date of the first review of the product / 7 months)
        """

        beta_3 = 0.69

        # for each product j, find the time of the earliest review F(j)
        first_time_product = {}
        for i, d in product_data.items():
            for t in d:
                if i not in first_time_product:
                    first_time_product[i] = datetime.strptime(t[3], self.date_time_format_str)
                elif datetime.strptime(t[3], self.date_time_format_str) < first_time_product[i]:
                    first_time_product[i] = datetime.strptime(t[3], self.date_time_format_str)

        etf = {}  # key = (u_id, p_id), value = maximum difference between reviews (u_id, p_id) and first review of the product
        for i, d in product_data.items():
            for t in d:
                td = datetime.strptime(t[3], self.date_time_format_str) - first_time_product[i]
                if (t[0], i) not in etf:
                    etf[(t[0], i)] = td
                # find the largest td for the review
                elif td > etf[(t[0], i)]:
                    etf[(t[0], i)] = td

        for k, v in etf.items():
            if v.days > 7 * 30:
                etf[k] = 0
            elif 1.0 - v.days / (7 * 30) > beta_3:
                etf[k] = 1
            else:
                etf[k] = 0
        return etf

    def iETF(self, product_data, new_product_data):
        beta_3 = 0.69
        # for each product j, find the time of the earliest review F(j)
        first_time_product = {}
        for i, d in new_product_data.items():
            all_d = deepcopy(d)
            if i in product_data:
                all_d = d + product_data[i]
            for t in all_d:
                if i not in first_time_product:
                    first_time_product[i] = datetime.strptime(t[3], self.date_time_format_str)
                elif datetime.strptime(t[3], self.date_time_format_str) < first_time_product[i]:
                    first_time_product[i] = datetime.strptime(t[3], self.date_time_format_str)

        etf = {}  # key = (u_id, p_id), value = maximum difference between reviews (u_id, p_id) and first review of the product
        for i, d in new_product_data.items():
            all_d = deepcopy(d)
            if i in product_data:
                all_d = d + product_data[i]
            for t in all_d:
                td = datetime.strptime(t[3], self.date_time_format_str) - first_time_product[i]
                if (t[0], i) not in etf:
                    etf[(t[0], i)] = td
                # find the largest td for the review
                elif td > etf[(t[0], i)]:
                    etf[(t[0], i)] = td

        for k, v in etf.items():
            if v.days > 7 * 30:
                etf[k] = 0
            elif 1.0 - v.days / (7 * 30) > beta_3:
                etf[k] = 1
            else:
                etf[k] = 0
        return etf

    def ISR(self, user_data):
        """
            Check if a user posts only one review
        """
        isr = {}
        for i, d in user_data.items():
            # go through all review of this user
            for t in d:
                if len(d) == 1:
                    isr[(i, t[0])] = 1
                else:
                    isr[(i, t[0])] = 0
        return isr

    def iISR(self, user_data, new_user_data):
        isr = {}
        for i, d in new_user_data.items():
            all_d = deepcopy(d)
            if i in user_data:
                all_d = d + user_data[i]
            # go through all review of this user
            for t in all_d:
                if len(all_d) == 1:
                    isr[(i, t[0])] = 1
                else:
                    isr[(i, t[0])] = 0
        return isr

    def add_feature(self, existing_features, new_features, feature_names):
        """
            Add or update feature(s) of a set of nodes of the same type to the existing feature(s).
            If a feature of a node is already is existing_features, then the new values will replace the existing ones.
            Args:
                existing_features: a dictionary {node_id:dict{feature_name:feature_value}}
                new_features: new feature(s) to be added. A dict {node_id: list of feature values}
                feature_names: the name of the new feature. A list of feature names, in the same order of the list of feature values in new_features
        """

        for k, v in new_features.items():
            # k is the node id and v is the feature value
            if k not in existing_features:
                existing_features[k] = dict()
            # add the new feature to the dict of the node
            for i in range(len(feature_names)):
                if len(feature_names) > 1:
                    existing_features[k][feature_names[i]] = v[i]
                else:
                    existing_features[k][feature_names[i]] = v

    def construct_all_features(self, user_data, prod_data):
        """
            Main entry to feature construction.
            Args:
                metadata_filename:
                text_feature_filename:
            Return:
                user, product and review features
        """

        # key = user id, value = dict of {feature_name: feature_value}
        UserFeatures = {}
        # key = product id, value = dict of {feature_name: feature_value}
        ProdFeatures = {}

        # go through feature functions
        # print ('\nadding user and product features......\n')
        # new feature
        uf = self.MNR(user_data, data_type='user')
        self.add_feature(UserFeatures, uf, ["MNR"])
        pf = self.MNR(prod_data, data_type='prod')
        self.add_feature(ProdFeatures, pf, ["MNR"])

        uf = self.PR_NR(user_data)
        self.add_feature(UserFeatures, uf, ["PR", "NR"])
        pf = self.PR_NR(prod_data)
        self.add_feature(ProdFeatures, pf, ["PR", "NR"])

        uf = self.avgRD_user(user_data, prod_data)
        self.add_feature(UserFeatures, uf, ["avgRD"])
        pf = self.avgRD_prod(prod_data)
        self.add_feature(ProdFeatures, pf, ["avgRD"])
        # new feature
        uf = self.BST(user_data)
        self.add_feature(UserFeatures, uf, ["BST"])

        uf = self.ERD(user_data)
        self.add_feature(UserFeatures, uf, ["ERD"])
        pf = self.ERD(prod_data)
        self.add_feature(ProdFeatures, pf, ["ERD"])
        # new feature
        uf = self.ETG(user_data)
        self.add_feature(UserFeatures, uf, ["ETG"])
        pf = self.ETG(prod_data)
        self.add_feature(ProdFeatures, pf, ["ETG"])

        # go through review features
        # print ('\nadding review features......\n')
        ReviewFeatures = {}
        rf = self.RD(prod_data)
        self.add_feature(ReviewFeatures, rf, ['RD'])

        rf = self.EXT(prod_data)
        self.add_feature(ReviewFeatures, rf, ['EXT'])

        rf = self.DEV(prod_data)
        self.add_feature(ReviewFeatures, rf, ['DEV'])

        rf = self.ETF(prod_data)
        self.add_feature(ReviewFeatures, rf, ['ETF'])

        rf = self.ISR(user_data)
        self.add_feature(ReviewFeatures, rf, ['ISR'])

        return UserFeatures, ProdFeatures, ReviewFeatures

    def update_all_features(self, user_data, new_user_data, prod_data, new_product_data, UserFeatures, ProdFeatures,
                            ReviewFeatures):
        """ Construct features using the new data (new_user_data, new_product_data) and update them to UserFeatures, ProdFeatures and ReviewFeatures
        """
        # go through feature functions
        uf = self.iMNR(user_data, new_user_data, data_type='user')
        self.add_feature(UserFeatures, uf, ["MNR"])
        pf = self.iMNR(prod_data, new_product_data, data_type='prod')
        self.add_feature(ProdFeatures, pf, ["MNR"])

        uf = self.iPR_NR(user_data, new_user_data)
        self.add_feature(UserFeatures, uf, ["PR", "NR"])
        pf = self.iPR_NR(prod_data, new_product_data)
        self.add_feature(ProdFeatures, pf, ["PR", "NR"])

        uf = self.iavgRD_user(user_data, new_user_data, prod_data, new_product_data)
        # assert ('201' in uf), 'user 201 not in feature avgRD_user'
        self.add_feature(UserFeatures, uf, ["avgRD"])
        # assert ('avgRD' in UserFeatures['201']), 'avgRD not in the features of user 201'
        pf = self.iavgRD_prod(prod_data, new_product_data)
        self.add_feature(ProdFeatures, pf, ["avgRD"])

        uf = self.iBST(user_data, new_user_data)
        self.add_feature(UserFeatures, uf, ["BST"])

        uf = self.iERD(user_data, new_user_data)
        self.add_feature(UserFeatures, uf, ["ERD"])
        pf = self.iERD(prod_data, new_product_data)
        self.add_feature(ProdFeatures, pf, ["ERD"])

        uf = self.iETG(user_data, new_user_data)
        self.add_feature(UserFeatures, uf, ["ETG"])
        pf = self.iETG(prod_data, new_product_data)
        self.add_feature(ProdFeatures, pf, ["ETG"])

        rf = self.iRD(prod_data, new_product_data)
        self.add_feature(ReviewFeatures, rf, ['RD'])

        rf = self.iEXT(prod_data, new_product_data)
        self.add_feature(ReviewFeatures, rf, ['EXT'])

        rf = self.iDEV(prod_data, new_product_data)
        self.add_feature(ReviewFeatures, rf, ['DEV'])

        rf = self.iETF(prod_data, new_product_data)
        self.add_feature(ReviewFeatures, rf, ['ETF'])

        rf = self.iISR(user_data, new_user_data)
        self.add_feature(ReviewFeatures, rf, ['ISR'])

        return UserFeatures, ProdFeatures, ReviewFeatures

    def calculateNodePriors(self, feature_names, features_py, when_suspicious):
        """
            Calculate priors of nodes P(y=1|node) using node features.
            Args:
                feature_names: a list of feature names for a particular node type.
                features_py: a dictionary with key = node_id and value = dict of feature_name:feature_value
                when_suspicious: a dictionary with key = feature name and value = 'H' (the higher the more suspicious) or 'L' (the opposite)
            Return:
                A dictionary with key = node_id and value = S score (see the SpEagle paper for the definition)
        """

        priors = {}
        for node_id, v in features_py.items():
            priors[node_id] = 0

        for f_idx, fn in enumerate(feature_names):

            fv_py = []
            for node_id, v in features_py.items():
                if fn not in v:
                    fv_py.append((node_id, -1))
                else:
                    fv_py.append((node_id, v[fn]))
            fv_py = sorted(fv_py, key=lambda x: x[1])

            i = 0
            while i < len(fv_py):
                start = i
                end = i + 1
                while end < len(fv_py) and fv_py[start][1] == fv_py[end][1]:
                    end += 1
                i = end

                for j in range(start, end):
                    node_id = fv_py[j][0]
                    if fv_py[j][0] == -1:
                        priors[node_id] += pow(0.5, 2)
                        continue
                    if when_suspicious[fn] == '+1':
                        priors[node_id] += pow((1.0 - float(start + 1) / len(fv_py)), 2)
                    else:
                        priors[node_id] += pow(float(end) / len(fv_py), 2)

        for node_id, v in features_py.items():
            priors[node_id] = 1.0 - math.sqrt(priors[node_id] / len(feature_names))
            if priors[node_id] > 0.999:
                priors[node_id] = 0.999
            elif priors[node_id] < 0.001:
                priors[node_id] = 0.001
        return priors

    def calNewNodePriors(self, feature_names, features_py, when_suspicious):
        """
            Calculate priors of nodes P(y=1|node) using node features.
            Args:
                feature_names: a list of feature names for a particular node type.
                features_py: a dictionary with key = node_id and value = dict of feature_name:feature_value
                when_suspicious: a dictionary with key = feature name and value = 'H' (the higher the more suspicious) or 'L' (the opposite)
            Return:
                A dictionary with key = node_id and value = S score (see the SpEagle paper for the definition)
        """

        priors = {}
        for node_id, v in features_py.items():
            priors[node_id] = []

        for f_idx, fn in enumerate(feature_names):

            fv_py = []
            for node_id, v in features_py.items():
                if fn not in v:
                    fv_py.append((node_id, -1))
                else:
                    fv_py.append((node_id, v[fn]))
            fv_py = sorted(fv_py, key=lambda x: x[1])

            i = 0
            while i < len(fv_py):
                start = i
                end = i + 1
                while end < len(fv_py) and fv_py[start][1] == fv_py[end][1]:
                    end += 1
                i = end

                for j in range(start, end):
                    node_id = fv_py[j][0]

                    if fv_py[j][0] == -1:
                        priors[node_id] += pow(0.5, 2)
                        continue
                    if when_suspicious[fn] == '+1':
                        priors[node_id].append((1.0 - float(start + 1) / len(fv_py)))
                        if node_id == '201':
                            print(priors[node_id])
                    else:
                        priors[node_id].append(float(end) / len(fv_py))
                        if node_id == '201':
                            print(priors[node_id])

        for node_id, v in features_py.items():
            # if int(node_id) <= 210:
            #     print(node_id)
            #     print(priors[node_id])
            priors[node_id] = min(priors[node_id])
            # if int(node_id) <= 210:
            #     print(priors[node_id])
            if priors[node_id] > 0.999:
                priors[node_id] = 0.999
            elif priors[node_id] < 0.001:
                priors[node_id] = 0.001
        return priors

if __name__ == '__main__':
    # path to the folder containing the files
    prefix = '../../Yelp_Dataset/YelpZip/'

    UserPriors = {}
    ProdPriors = {}
    ReviewPriors = {}

    # raw data file names
    metadata_filename = prefix + 'metadata.gz'
    review_filename = prefix + 'reviewContent'

    # feature file names
    user_feature_filename = prefix + 'UserFeatures.pickle'
    prod_feature_filename = prefix + 'ProdFeatures.pickle'
    review_feature_filename = prefix + 'ReviewFeatures.pickle'

    # model file names
    prod_model = prefix + 'prod_model.pkl'
    user_model = prefix + 'user_model.pkl'
    review_model = prefix + 'review_model.pkl'

    # prior file names
    user_prior_filename = prefix + 'UserPriors.pickle'
    prod_prior_filename = prefix + 'ProdPriors.pickle'
    review_prior_filename = prefix + 'ReviewPriors.pickle'

    # feature configuration
    feature_suspicious_filename = 'feature_configuration.txt'

    # all high level features
    print('Starting constructing high level user, product and review features\n')
    user_data, prod_data = read_graph_data(metadata_filename)

    UserFeatures, ProdFeatures, ReviewFeatures = construct_all_features(user_data, prod_data)

    with open(user_feature_filename, 'wb') as f:
        pickle.dump(UserFeatures, f)

    with open(prod_feature_filename, 'wb') as f:
        pickle.dump(ProdFeatures, f)

    with open(review_feature_filename, 'wb') as f:
        pickle.dump(ReviewFeatures, f)
    print('Finished constructing high level user, product and review features\n')
