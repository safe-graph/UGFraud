from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import gzip
import numpy as np
import networkx as nx
import time
import functools
import warnings


def create_ground_truth(user_data):
    """Given user data, return a dictionary of labels of users and reviews
    Args:
        user_data: key = user_id, value = list of review tuples.
    Return:
        user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
        review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
    """
    user_ground_truth = {}
    review_ground_truth = {}

    for user_id, reviews in user_data.items():

        user_ground_truth[user_id] = 0

        for r in reviews:
            prod_id = r[0]
            label = r[2]

            if label == -1:
                review_ground_truth[(user_id, prod_id)] = 1
                user_ground_truth[user_id] = 1
            else:
                review_ground_truth[(user_id, prod_id)] = 0

    return user_ground_truth, review_ground_truth


def evaluate(y, pred_y):
    """
    Revise: test when a key is a review/account.
    Evaluate the prediction of account and review by SpEagle
    Args:
        y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)
        pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
                the keys in pred_y must be a subset of the keys in y
    """
    posteriors = []
    ground_truth = []

    for k, v in pred_y.items():
        if k in y:
            posteriors.append(v)
            ground_truth.append(y[k])

    if len(np.unique(ground_truth)) < 2:
        warnings.warn("Only one class present in ground_truth, ROC AUC score will be omitted")
        ap = average_precision_score(ground_truth, posteriors)
        return None, ap
    else:
        auc = roc_auc_score(ground_truth, posteriors)
        ap = average_precision_score(ground_truth, posteriors)
        return auc, ap


def scale_value(value_dict):
    """
    Calculate and return a dict of the value of input dict scaled to (0, 1)
    """

    ranked_dict = [(user, value_dict[user]) for user in value_dict.keys()]
    ranked_dict = sorted(ranked_dict, reverse=True, key=lambda x: x[1])

    up_max, up_mean, up_min = ranked_dict[0][1], ranked_dict[int(len(ranked_dict) / 2)][1], ranked_dict[-1][1]

    scale_dict = {}
    for i, p in value_dict.items():
        norm_value = (p - up_min) / (up_max - up_min)
        if norm_value == 0:  # avoid the 0
            scale_dict[i] = 0 + 1e-7
        elif norm_value == 1:  # avoid the 1
            scale_dict[i] = 1 - 1e-7
    else:
        scale_dict[i] = norm_value

    return scale_dict


def nor_priors(priors):
    """
    Normalize the node priors for GANG
    :param priors:
    :return:
    """
    new_upriors, new_rpriors, new_ppriors = priors

    # normalize the node priors to (0,1)
    # if we normalize the prior, we need to set nor_flg to True for the gang model
    ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
    ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
    ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
    ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
    ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
    ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
    u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
    p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
    r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]
    for i, p in priors[0].items():
        priors[0][i] = (p - u_min) / (u_max - u_min)
    for i, p in priors[1].items():
        priors[1][i] = (p - r_min) / (r_max - r_min)
    for i, p in priors[2].items():
        priors[2][i] = (p - p_min) / (p_max - p_min)

    return priors, [u_mean, r_mean, p_mean]


def get_hash(data):
    import hashlib
    return hashlib.md5(data).hexdigest()


def read_graph_data(metadata_filename, adj=False):
    """ Read the user-review-product graph from file. Can output the graph in different formats
        Args:
            metadata_filename: a gzipped file containing the graph.
            adj: if True: create adjacent data, default is False
        Return:
            graph: user-review / prod-review / list of adjacent(adj=True)
    """

    user_data = {}

    prod_data = {}

    adj_data = []

    # use the rt mode to read ascii strings instead of binary
    if adj is False:
        with gzip.open(metadata_filename, 'rt') as f:
            # file format: each line is a tuple (user id, product id, rating, label, date)
            for line in f:
                items = line.strip().split()
                u_id = items[0]
                p_id = items[1]
                if items[2] != 'None':
                    rating = float(items[2])
                else:
                    rating = 'None'
                label = int(items[3])
                date = items[4]

                if u_id not in user_data:
                    user_data[u_id] = []
                user_data[u_id].append((p_id, rating, label, date))

                if p_id not in prod_data:
                    prod_data[p_id] = []
                prod_data[p_id].append((u_id, rating, label, date))

            # create adj_list [u_id, p_id, 1/2], where 1 indicates positive rating (4, 5)
            # and 2 indicates negative rating (1, 2, 3)

        print('read reviews from %s' % metadata_filename)
        print('number of users = %d' % len(user_data))
        print('number of products = %d' % len(prod_data))
        return user_data, prod_data
    else:
        # create adj_list [u_id, p_id, 1/2], where 1 indicates positive rating (4, 5)
        # and 2 indicates negative rating (1, 2, 3)
        with gzip.open(metadata_filename, 'rt') as f:
            # file format: each line is a tuple (user id, product id, rating, label, date)
            for line in f:
                items = line.strip().split()
                u_id = items[0]
                p_id = items[1]
                if items[2] != 'None':
                    rating = float(items[2])
                else:
                    rating = 'None'
                label = int(items[3])
                date = items[4]

                if u_id not in user_data:
                    user_data[u_id] = []
                user_data[u_id].append((p_id, rating, label, date))

                if p_id not in prod_data:
                    prod_data[p_id] = []
                prod_data[p_id].append((u_id, rating, label, date))

                if int(rating) <= 3:
                    rating = int(2)
                else:
                    rating = int(1)
                adj_data.append([u_id, p_id, rating])

        print('read reviews from %s' % metadata_filename)
        print('number of users = %d' % len(user_data))
        print('number of products = %d' % len(prod_data))
        print('number of ratings = %d' % len(adj_data))
        return user_data, prod_data, np.array(adj_data, dtype='int32')


def depth(data):
    """
    Get the depth of a dictionary
    Args:
        data: data in dictionary type

    Returns: the depth of a dictionary

    """
    if isinstance(data, dict):
        return 1 + (max(map(depth, data.values())) if data else 0)
    return 0


def data_checker(data):
    """
    data validation
    Args:
        data: data in dictionary type

    Returns: pass the validation

    """
    if isinstance(data, dict):
        if depth(data) < 3:
            raise Exception("The minimum depth of data must be 3. For example: {\'node1\':{\'node1_neighbor\':{"
                            "neighbor's attribute}}}")
    else:
        raise AttributeError("Data must be stored in dictionary.")


def dict_to_networkx(data):
    """
    Convert data into networkx graph
    Args:
        data: data in dictionary type

    Returns: networkx graph

    """
    data_checker(data)
    G = nx.Graph(data)
    return G


def add_attribute_to_graph(graph, attribute, adding_type):
    """
    Add new attributes to nodes/edges
    Args:
        graph: networkx graph
        attribute: dictionary of attributes for nodes/edges
        adding_type: string of node or edge

    Returns:
        networkx graph with new attributes
    """
    if isinstance(attribute, dict):
        if isinstance(graph, nx.classes.graph.Graph):
            if adding_type == 'node':
                nx.set_node_attributes(graph, attribute)
                return graph
            elif adding_type == 'edge':
                nx.set_edge_attributes(graph, attribute)
                return graph
            else:
                raise Exception("Adding type must be \'node\' or \'edge\'.")
        else:
            raise Exception("The graph must be a networkx graph.")
    else:
        raise AttributeError("Attribute must be stored in dictionary.")


def get_node_attributes_index(graph, attr):
    """
    get node index for each attributes
    Args:
        graph: networkx graph
        attr: nodes' attribute

    Returns:
        a dict of list which contains every attribute index
        For example: {'user': ['201','202','203','204'], 'prod': ['0', '1', '2']}
    """
    from collections import defaultdict
    node_temp = nx.get_node_attributes(graph, attr)
    reversed_dict = defaultdict(list)
    for key, value in node_temp.items():
        reversed_dict[value].append(key)
    return reversed_dict


def get_edge_attributes_index(graph, attr):
    """
    get edge index for each attributes
    Args:
        graph: networkx graph
        attr: edges' attribute

    Returns:
        a dict of list which contains every attribute index
        For example: {'review': [('201', '0'), ('202', '0'), ('203', '0'), ('204', '0')]}
    """
    from collections import defaultdict
    node_temp = nx.get_edge_attributes(graph, attr)
    reversed_dict = defaultdict(list)
    for key, value in node_temp.items():
        reversed_dict[value].append(key)
    return reversed_dict


def node_attr_filter(graph, attr, specific_attr, into_attr):
    """
    get specific keys, values in conditions
    Args:
        graph: networkx graph
        attr: which attribute index you want to get
        specific_attr: which specific attribute index you want to get depending on attr
        into_attr: use specific attribute index to filter the attribute

    Returns:
        dict(node: into_attr values)
        For example: node_attr_filter(graph, 'types', 'user', 'prior)
        will return the dict( user_id: user_id_prior)

    """
    attr_dict_index = get_node_attributes_index(graph, attr)
    specific_dict = attr_dict_index[specific_attr]
    filtered_dict = dict()
    into_dict = nx.get_node_attributes(graph, into_attr)
    for i in specific_dict:
        filtered_dict[i] = into_dict[i]
    return filtered_dict


def edge_attr_filter(graph, attr, specific_attr, into_attr):
    """
    get specific keys, values in conditions
    Args:
        graph: networkx graph
        attr: which attribute index you want to get
        specific_attr: which specific attribute index you want to get depending on attr
        into_attr: use specific attribute index to filter the attribute

    Returns:
        dict(edge: into_attr values)
        For example: edge_attr_filter(graph, 'types', 'review', 'prior)
        will return the dict(review_id: review_id_prior)

    """
    attr_dict_index = get_edge_attributes_index(graph, attr)
    specific_dict = attr_dict_index[specific_attr]
    filtered_dict = dict()
    into_dict = nx.get_edge_attributes(graph, into_attr)
    for i in specific_dict:
        filtered_dict[i] = into_dict[i]
    return filtered_dict


def save_graph(graph, graph_name=False):
    """

    Args:
        graph: network graph
        graph_name: the file name of the graph, if graph_name=False, use default name

    Returns:
        None
    """
    from networkx.readwrite import json_graph
    import json
    data = json_graph.node_link_data(graph)
    if graph_name is False:
        graph_name = 'graph_data.json'
    with open(graph_name, 'w') as f:
        json.dump(data, f)
    f.close()
    print('Saved graph data as {}'.format(graph_name))


def load_graph(json_name):
    """

    Args:
        json_name: json file name

    Returns:
        networkx graph
    """
    from networkx.readwrite import json_graph
    import json
    with open(json_name, 'r') as f:
        data = json.load(f)
    f.close()
    graph = json_graph.node_link_graph(data)
    print('Loaded {} into the nextorkx graph'.format(json_name))
    return graph


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(func.__name__, round(run_time, 3)))
        return value
    return wrapper_timer

