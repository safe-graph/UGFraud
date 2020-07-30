"""
    ZooBP: Belief Propagation for Heterogeneous Networks.
    A method to perform fast BP on undirected heterogeneous graphs with provable convergence guarantees.
    Article: http://www.vldb.org/pvldb/vol10/p625-eswaran.pdf
"""

from UGFraud.Utils.helper import timer
from scipy.special import logsumexp
from scipy import sparse
from collections import defaultdict
import numpy as np
import networkx as nx


def Initialize_Final_Beliefs(N1, N2, m):
    """
        Initialization of final beliefs
        Args:
            N1: number of users
            N2: number of products
            m: coefficient for reduction in beliefs
        Returns:
            Concatenation of initialized final beliefs for users and products
        Example of return values: -0.5 0.5 -0.3 0.3 ...
    """
    r1 = m * (np.random.uniform(size=N1) - 0.5)
    r1 = r1.reshape(r1.shape[0], 1)
    r2 = m * (np.random.uniform(size=N2) - 0.5)
    r2 = r2.reshape(r2.shape[0], 1)
    B1 = np.concatenate((r1, -r1), axis=1)
    B2 = np.concatenate((r2, -r2), axis=1)

    temp1_B = B1.reshape((B1.shape[1] * B1.shape[0], 1))
    temp2_B = B2.reshape((B2.shape[1] * B2.shape[0], 1))
    B = np.concatenate((temp1_B, temp2_B), axis=0)

    return B


class ZooBP:
    def __init__(self, graph, ep, H):
        """
            implementation of ZooBP in python
            Args:
                graph: a networkx graph
                ep: interaction strength
                H: compatibility matrix
            Returns:
                final_user_beliefs: centered version of final user beliefs
                final_prod_beliefs centered version of final prod beliefs
            NOTE:
                ZooBP requires consecutive ids not ids with gaps
        """
        a_list_temp = nx.get_edge_attributes(graph, 'rating')
        n, p = list(zip(*list(a_list_temp.keys())))
        reversed_dict = defaultdict(list)
        node_types_index = nx.get_node_attributes(graph, 'types')
        for key, value in node_types_index.items():
            reversed_dict[value].append(key)
        self.a_list = np.array(list(zip(n, p, a_list_temp.values())), dtype=np.int32)
        u_priors = dict()
        p_priors = dict()
        node_prior_index = nx.get_node_attributes(graph, 'prior')
        for i in reversed_dict['user']:
            u_priors[i] = node_prior_index[i]
        for i in reversed_dict['prod']:
            p_priors[i] = node_prior_index[i]
        self.u_tag, user_priors = zip(*u_priors.items())
        self.u_priors = np.array(user_priors)
        self.p_tag, prod_priors = zip(*p_priors.items())
        self.p_priors = np.array(prod_priors)
        self.ep = ep
        self.H = H

    @timer
    def run(self):
        # converts the given priors to the centered version
        user_priors = self.u_priors - 0.5 * np.ones((self.u_priors.shape[0]))
        prod_priors = self.p_priors - 0.5 * np.ones((self.p_priors.shape[0]))
        # finds positive (1) and negative (2) edges and reshapes them
        rating = self.a_list[:, 2]
        self.a_list[self.a_list[:, 2] == 2] = 2
        self.a_list[self.a_list[:, 2] == 1] = 1
        edges_pos = self.a_list[rating == 1]
        edges_neg = self.a_list[rating == 2]
        Lpos = edges_pos[:, 0:2]
        Lpos = Lpos.reshape((edges_pos.shape[0], 2))
        Lneg = edges_neg[:, 0:2]
        Lneg = Lneg.reshape((edges_neg.shape[0], 2))
        n_user = user_priors.shape[0]
        n_prod = prod_priors.shape[0]

        # computes A+ and A- as defined in section 4.7 of ZooBP
        lpos_0 = Lpos[:, 0] - np.ones(Lpos[:, 0].shape[0])
        lpos_1 = Lpos[:, 1] - np.ones(Lpos[:, 1].shape[0])
        Apos = sparse.coo_matrix((np.ones(Lpos.shape[0]), (lpos_0, lpos_1)), shape=(n_user, n_prod))
        lneg_0 = Lneg[:, 0] - np.ones(Lneg[:, 0].shape[0])
        lneg_1 = Lneg[:, 1] - np.ones(Lneg[:, 1].shape[0])
        Aneg = sparse.coo_matrix((np.ones(len(Lneg)), (lneg_0, lneg_1)), shape=(n_user, n_prod))

        # prior beliefs are reshaped so that user1_belief 1-user1_belief ... prod1_belief 1-prod1_belief
        h_user_priors = np.reshape(user_priors, (len(user_priors), -1))
        h_prod_priors = np.reshape(prod_priors, (len(prod_priors), -1))
        user_priors = np.hstack((h_user_priors, -h_user_priors))
        prod_priors = np.hstack((h_prod_priors, -h_prod_priors))
        reshape_u = user_priors.reshape((2 * n_user, 1))
        reshape_p = prod_priors.reshape((2 * n_prod, 1))
        E = np.concatenate((reshape_u, reshape_p))

        # build P defined under section 4.7 of ZooBP
        R = sparse.kron(Apos - Aneg, self.ep * self.H)
        sp1 = sparse.coo_matrix((2 * n_user, 2 * n_user), dtype=np.int8)
        temp1 = sparse.hstack([sp1, 0.5 * R])
        sp2 = sparse.coo_matrix((2 * n_prod, 2 * n_prod), dtype=np.int8)
        temp2 = sparse.hstack([0.5 * R.transpose(), sp2])
        P = sparse.vstack((temp1, temp2))
        P = P.transpose()

        # build Q defined under section 4.7 of ZooBP
        sum_temp = Apos + Aneg
        temp1 = sum_temp.sum(axis=1)
        temp2 = sum_temp.sum(axis=0)
        D12 = sparse.diags(np.asarray(temp1.flatten()).reshape(-1))
        D21 = sparse.diags(np.asarray(temp2.flatten()).reshape(-1))
        temp = 0.25 * self.ep * self.ep * sparse.kron(D12, self.H)
        Q_1 = sparse.eye(n_user * 2) + temp
        Q_2 = sparse.eye(n_prod * 2) + (0.25 * self.ep * self.ep) * (sparse.kron(D21, self.H))
        sp1 = sparse.coo_matrix((n_user * 2, n_prod * 2), dtype=np.int8)
        Q_temp1 = sparse.hstack((Q_1, sp1))
        sp2 = sparse.coo_matrix((n_prod * 2, n_user * 2), dtype=np.int8)
        Q_temp2 = sparse.hstack((sp2, Q_2))
        Q = sparse.vstack((Q_temp1, Q_temp2))

        # M
        M = P - Q + sparse.eye(2 * (n_user + n_prod))
        M = M.transpose()
        B = Initialize_Final_Beliefs(n_user, n_prod, 0.001)

        # Iterative Solution
        res = 1
        while (res > 1e-8):
            Bold = B
            # Equations (13) and (14) in ZooBP
            B = E + logsumexp(M * Bold)
            res = np.sum(np.sum(abs(Bold - B)))

        B1 = B[0:2 * n_user, :]
        B2 = B[2 * n_user:, :]
        user_beliefs = B1.reshape((n_user, 2))
        user_beliefs = dict(zip(self.u_tag, user_beliefs[:, 0]))
        prod_beliefs = B2.reshape((n_prod, 2))
        prod_beliefs = dict(zip(self.p_tag, prod_beliefs[:, 0]))

        return user_beliefs, prod_beliefs





