import gzip
import pickle
import random

"""
	Define several functions to handle files.
"""


def load_text_feature(filename):
	with open(filename, 'rb') as in_f:
		review_text_feature = pickle.load(in_f)
	return review_text_feature


def read_graph_data(metadata_filename, training=True):
	""" Read the user-review-product graph from file. Can output the graph in different formats
		Args:
			metadata_filename: a gzipped file containing the graph.
			training: training flag
		Return:
			graph: user-review / prod-review
	"""

	user_data = {}

	prod_data = {}

	# use the rt mode to read ascii strings instead of binary
	with gzip.open(metadata_filename, 'rt') as f:
		# file format: each line is a tuple (user id, product id, rating, label, date)
		for line in f:
			items = line.strip().split()
			u_id = items[0]
			p_id = items[1]
			rating = float(items[2])
			label = int(items[3])
			date = items[4]

			if u_id not in user_data:
				user_data[u_id] = []
			user_data[u_id].append((p_id, rating, label, date))

			if p_id not in prod_data:
				prod_data[p_id] = []
			prod_data[p_id].append((u_id, rating, label, date))

	# delete reviews from graph to create new products
	if 'NYC' in metadata_filename and training:
		user_data, prod_data = create_new_products(user_data, prod_data, (0, 120))
	if 'NYC' in metadata_filename and not training:
		user_data, prod_data = create_new_products(user_data, prod_data, (120, 240))
	if 'Chi' in metadata_filename and not training:
		user_data, prod_data = create_new_products(user_data, prod_data, (30, 60))

	# print('read reviews from %s' % metadata_filename)
	# print('number of users = %d' % len(user_data))
	# print('number of products = %d' % len(prod_data))

	return user_data, prod_data


def load_feature_names(path, filename):
	"""
		Read names of features for each of the 3 different kinds of nodes.
		Note that no polarity is recorded here.
	"""
	user_feature_names = []
	prod_feature_names = []
	review_feature_names = []

	# with open('../SpEagle/featureExtractorMatlab/feature_list_matlab.txt', 'r') as f:
	with open(path + filename, 'r') as f:
		for line in f:
			if line[0] == ';' and 'user' in line:
				line = f.readline()
				user_feature_names = line.strip().split()
			elif line[0] == ';' and 'product' in line:
				line = f.readline()
				prod_feature_names = line.strip().split()
			elif line[0] == ';' and 'review' in line:
				line = f.readline()
				review_feature_names = line.strip().split()
	return user_feature_names, prod_feature_names, review_feature_names


def load_feature_config(config_filename):
	"""Read configuration about how the value of a feature indicates suspiciousness of a node
		
	"""
	config = {}

	f = open(config_filename, 'r')
	for line in f:
		if line[0] == '+' or line[0] == '-':
			# print (line)
			items = line.split(' ')
			direction = items[0].strip(':')
			feature_name = items[1]
			config[feature_name] = direction
	f.close()

	return config

def ConvertFromSpEagleToCrf(user_data, user_features, prod_features, review_features, crf_filename, output_labels = True, labeled_id_filename='labeled_ids.txt'):
	""" Convert graph and node features used in SpEagle to format that can be read by CRF
	id of a user: u100
	id of a product: p201
	id of a review: u100p201
	Args:
		output_labels: output a separate file containing the ids of labeled reviews if true
	"""
	user_labels = {}
	prod_labels = {}
	review_labels = {}

	for user_id, reviews in user_data.items():
		user_labels[user_id] = 0
		for r in reviews:
			prod_id = r[0]
			prod_labels[prod_id] = 0

			# label == -1 => spam
			if r[2] == -1:
				review_labels[(user_id, prod_id)] = 1
				user_labels[user_id] = 1
			else:
				review_labels[(user_id, prod_id)] = 0

	print('# users = %d\n# products = %d\n#reviews = %d' % (len(user_labels), len(prod_labels), len(review_labels)))

	with open(crf_filename, 'w') as f:
		for k, v in user_labels.items():
			# write user_id (prefixed by u) and label
			f.write('u%s %d ' % (k, v))

			# output user features (prefixed by u)
			for fn, fv in user_features[k].items():
				if fv != 0.0:
					f.write('u%s:%f ' % (fn, fv))
			f.write('\n')

		for k, v in prod_labels.items():
			# write prod_id (prefixed by p) and label
			f.write('p%s %d ' % (k, v))

			# output product features (prefixed by p)
			for fn, fv in prod_features[k].items():
				if fv != 0.0:
					f.write('p%s:%f ' % (fn, fv))
			f.write('\n')

		for k, v in review_labels.items():

			# write review_id and label
			f.write('u%s_p%s %d ' % (k[0], k[1], v))

			# output review features (prefixed by r)
			for fn, fv in review_features[k].items():
				if fv != 0.0:
					f.write('r%s:%f ' % (fn, fv))
			f.write('\n')

		# output edges
		# format: #edge <src_name> <dst_name> <edge_type>
		for user_id, reviews in user_data.items():
			for r in reviews:
				prod_id = r[0]
				# edge between user and the review
				f.write('#edge u%s u%s_p%s ur\n' % (user_id, user_id, prod_id))
				# edge between product and the review
				f.write('#edge p%s u%s_p%s pr\n' % (prod_id, user_id, prod_id))
	if output_labels:
		with open(labeled_id_filename, 'w') as f:
			num_labeled_reviews = int(0.1 * len(review_labels))
			print('# of reviews %d, labeled reviews %d' % (len(review_labels), num_labeled_reviews))
			labeled_reviews = random.sample(list(review_labels.keys()), num_labeled_reviews)
			num_pos = 0
			num_neg = 0
			for k in labeled_reviews:
				f.write('u%s_p%s\n' % (k[0], k[1]))
				if review_labels[k] == 0:
					num_neg += 1
				else:
					num_pos += 1
			print('# of labeled pos = %d' % num_pos)
			print('# of labeled neg = %d' % num_neg)

def create_new_products(user_data, prod_data, range):
	"""
	Special pre-processing for the YelpNYC and YelpChi when there is no new products (#reviews<10)
	:param range: the range of products to be processed
	"""

	product_list = [(product, len(user)) for (product, user) in prod_data.items()]
	sorted_product_list = sorted(product_list, reverse=False, key=lambda x: x[1])
	new_products = [product[0] for product in sorted_product_list[range[0]:range[1]]]
	for item in new_products:
		for review in prod_data[item][1:]:
			for r in user_data[review[0]]:
				if r[0] == item:
					user_data[review[0]].remove(r)
			if len(user_data[review[0]]) == 0:
				user_data.pop(review[0])
		prod_data[item] = [prod_data[item][0]]

	# renumbering the graphs
	start_no = len(prod_data)
	index = {}
	for user in user_data.keys():
		if user not in index.keys():
			index[user] = str(start_no)
			start_no += 1
	new_user_data = {}
	new_prod_data = {}
	for user, reviews in user_data.items():
		new_user_data[index[user]] = reviews
	for prod, reviews in prod_data.items():
		new_prod_data[prod] = []
		for review in reviews:
			new_prod_data[prod].append((index[review[0]], review[1], review[2], review[3]))

	return new_user_data, new_prod_data
