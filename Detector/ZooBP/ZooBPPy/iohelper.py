"""
	Define several functions to handle files.
"""

def load_graphs(filename):
	"""
		Read the graphs.
		Graph format:
			each line is a tuple (reviewer_id, product_id, rating), both ids start from 1.
		Map the (reviewer_id, product_id) pair to review id:
			review_id shall start from 0, and the row numbers in the graph file serve as the review id
	"""
	reviewer_review_graph = {}

	review_product_graph = {}

	all_products = set()

	f = open(filename, 'r')

	review_id = 0;

	for line in f:
		u_id, p_id, _ = line.strip().split(",")
		u_id = int(u_id) - 1
		p_id = int(p_id) - 1
		
		#print(u_id + " " + p_id)
		#break

		all_products.add(p_id)

		# add the review to the user: a user can have multiple reviews
		if u_id not in reviewer_review_graph:
			reviewer_review_graph[u_id] = []
		reviewer_review_graph[u_id].append(review_id)

		# add the product to the review: there is one and only one target product of a review
		review_product_graph[review_id] = p_id

		review_id += 1

	f.close()

	print("Number of reviewers: " + str(len(set(reviewer_review_graph.keys()))))
	print("Max reviewer id: " + str(max(reviewer_review_graph.keys())))
	print("Min reviewer id: " + str(min(reviewer_review_graph.keys())))
	print("Number of reviews: " + str(len(review_product_graph)))
	print("Number of products: " + str(len(all_products)))

	return reviewer_review_graph, review_product_graph

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
