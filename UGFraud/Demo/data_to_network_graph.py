from UGFraud.Utils.helper import *
import networkx as nx
import sys
import os
import random
import pickle as pkl
sys.path.insert(0, os.path.abspath('../../'))


def data_to_network_graph():
    # data source
    data_name = 'YelpChi'
    prefix = '../Yelp_Data/' + data_name + '/'
    metadata_filename = prefix + 'metadata.gz'
    Checksum = 'f454ce0a5f506e0be062dc8aefb76b25'
    AUTHORIZED = False

    # valid YelpChi data
    with gzip.open(metadata_filename, 'rb') as f:
        file_content = f.read()
        f.close()
    if Checksum == get_hash(file_content):
        AUTHORIZED = True
    else:
        print('-' * 80)
        print('The demo data is not the intact data, if you need intact data, please download from:')
        print('http://odds.cs.stonybrook.edu/yelpchi-dataset/')
        print('-' * 80)

    """ 
     read the graph and node priors
     user_product_graph: {'201': [('0', 1)], ... }
     product_user_graph: {'0': [('201', 1), ('202', 1), ...], ...}

    """
    user_product_graph, product_user_graph = read_graph_data(metadata_filename)
    user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

    # load priors
    with open(prefix + 'priors.pkl', 'rb') as f:
        priors = pkl.load(f)

    # convert user_product_graph to dict of dict
    # graph_dict: {'201': {'0': {'rating': 1, 'label': 1, 'date': '2011-06-08'}},...}
    graph_dict = dict()
    for k, v in user_product_graph.items():
        graph_dict[k] = dict()
        for line in v:
            if line[2] == -1:
                new_line_2 = 0
            else:
                new_line_2 = 1
            # if demo data is not intact, generate rating randomly
            if type(line[1]) is str:
                new_line_1 = random.choice([0, 1])
            elif line[1] >= 4:
                new_line_1 = 1
            else:
                new_line_1 = 2
            graph_dict[k][line[0]] = {'rating': new_line_1, 'label': new_line_2, 'date': line[3]}

    # put graph_dict into networkx graph
    G = dict_to_networkx(graph_dict)

    # we also can convert the graph into dict of dicts
    dict_of_dicts = nx.to_dict_of_dicts(G)

    # organize nodes' attributes, attributes must be the dict of dicts:
    # for example: {'201': {'prior': 0.1997974972380755, 'types': 'user'}, ...}
    user_node_priors = priors[0]
    node_attr = dict()
    for k, v in user_node_priors.items():
        node_attr[k] = {'prior': v, 'types': 'user', 'label': user_ground_truth[k]}
    # add nodes' new attributes to the graph
    add_attribute_to_graph(graph=G, attribute=node_attr, adding_type='node')
    prod_node_priors = priors[2]
    node_attr = dict()
    for k, v in prod_node_priors.items():
        node_attr[k] = {'prior': v, 'types': 'prod'}
    # add nodes' new attributes to the graph
    add_attribute_to_graph(graph=G, attribute=node_attr, adding_type='node')

    # check new attributes
    G.nodes.get('201')

    # organize edges'attributes, attributes must be the dict of dicts:
    # for example: {('201', '0'): {'prior': 0.35048557119705304, 'types': 'review'}, ...}
    edge_priors = priors[1]
    edge_attr = dict()
    for k, v in edge_priors.items():
        edge_attr[k] = {'prior': v, 'types': 'review'}
    # add edges' new attributes to the graph
    add_attribute_to_graph(graph=G, attribute=edge_attr, adding_type='edge')
    # check new attributes
    G.edges.get(('201', '0'))

    # save graph data into json
    graph_name = 'Yelp_graph_data.json'
    save_graph(graph=G, graph_name=graph_name)

    # load json into graph
    loaded_G = load_graph(graph_name)


if __name__ == '__main__':
    data_to_network_graph()
