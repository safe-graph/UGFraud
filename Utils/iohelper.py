import gzip

"""
	Define several functions to handle files.
"""


def read_graph_data(metadata_filename):
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

	print('read reviews from %s' % metadata_filename)
	print('number of users = %d' % len(user_data))
	print('number of products = %d' % len(prod_data))

	return user_data, prod_data
