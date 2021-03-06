"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf
"""
import sys
import pickle
from time import time
from normalize import scale
from utils import dict_to_graph
from utils import add_shortest_path
from utils import get_edges_from_list
import numpy as np

if len(sys.argv) < 3:
	print("Provide a file name and a normalization method")
	exit()

#Import data

try:
	fname = sys.argv[1]
	f = open(fname)
	lines = list(f)
	edges      = []
	senders    = []
	receivers  = []
	lastNodeID = -1
	normMethod = int(sys.argv[2])

	for line in lines:
		comps = line.strip().split(' ')
		edges.append(float(comps[3]))
		senders.append(int(comps[1]))
		receivers.append(int(comps[2]))

		if int(comps[1]) > lastNodeID:
			lastNodeID = int(comps[1])
		if int(comps[2]) > lastNodeID: 
			lastNodeID = int(comps[2])


		#Just for test
		#if comps[0] == '10':
		#	break

	
	nodes = list(range(lastNodeID+1))
	# Scale the data
	edges = scale(edges, normMethod)
	
	edges = get_edges_from_list(edges)
	graph = {
	    "nodes": nodes,
	    "edges": edges,
	    "senders": senders,
	    "receivers": receivers
	}

	seed = 2
	rand = np.random.RandomState(seed=seed)
	print("Convert dictionary to graph")
	graph = dict_to_graph(graph)
	print("Shortest path")
	graph = add_shortest_path(rand, graph)

	#save data to file
	if len(sys.argv) > 3:
		pref = sys.argv[3]+"_"
	else:
		pref = "graph_"+str(int(time()))+"_nor_"+str(normMethod)+"_"

	with open(pref+fname+".out", 'wb') as fp:
		pickle.dump(graph, fp) 

finally:
	f.close();

