#!/usr/bin/env python3
from __future__ import division
from math import log, sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import sys
import os
from load_map import *
STEMMER = PorterStemmer()
import numpy as np
from scipy.stats import *
# helper function to get the count of a word (string)
def w_count(word):
	return o_counts[word2wid[word]]


def tw_stemmer(word):
	'''Stems the word using Porter stemmer, unless it is a
	username (starts with @).  If so, returns the word unchanged.

	:type word: str
	:param word: the word to be stemmed
	:rtype: str
	:return: the stemmed word

	'''
	if word[0] == '@':  # don't stem these
		return word
	else:
		return STEMMER.stem(word)


def PMI(c_xy, c_x, c_y, N, args = None):
	'''Compute the pointwise mutual information using cooccurrence counts.

	:type c_xy: int
	:type c_x: int
	:type c_y: int
	:type N: int
	:param c_xy: coocurrence count of x and y
	:param c_x: occurrence count of x
	:param c_y: occurrence count of y
	:param N: total observation count
	:rtype: float
	:return: the pmi value

	'''
	return log(((N * c_xy) / (c_x * c_y)), 2)

def ttest(c_xy, c_x, c_y, N, args = None):
	'''Compute ttest using cooccurrence counts.

	:type c_xy: int
	:type c_x: int
	:type c_y: int
	:type N: int
	:param c_xy: coocurrence count of x and y
	:param c_x: occurrence count of x
	:param c_y: occurrence count of y
	:param N: total observation count
	:rtype: float
	:return: the ttest value
	'''
	return (c_xy - c_x * c_y/N) / sqrt(c_x * c_y)
def PMI_alpha(c_xy, c_x, c_y, N, args):
	'''Compute the smoothed pointwise mutual information using cooccurrence counts.

	:type c_xy: int
	:type c_x: int
	:type c_y: int
	:type N: int
	:type args: tuple
	:param c_xy: coocurrence count of x and y
	:param c_x: occurrence count of x
	:param c_y: occurrence count of y
	:param N: total observation count
	:param args: (None, alpha value)
	:rtype: float
	:return: the smoothed pmi value

	'''
	alpha = args[1]
	return log(c_xy / (c_x * (c_y ** alpha / N ** alpha)), 2)

def cos_sim(v0, v1):
	'''Compute the cosine similarity between two sparse vectors.

	:type v0: dict
	:type v1: dict
	:param v0: first sparse vector
	:param v1: second sparse vector
	:rtype: float
	:return: cosine between v0 and v1
	'''
	# We recommend that you store the sparse vectors as dictionaries
	# with keys giving the indices of the non-zero entries, and values
	# giving the values at those dimensions.

	# You will need to replace with the real function

	# sparse vector to dense vector

	common_keys = v0.keys() & v1.keys()  # take the intersection of two list of keys
	numerator = sum([v0[k] * v1[k] for k in common_keys])

	v0_len = np.sqrt(np.sum(np.array(list(v0.values())) ** 2))
	v1_len = np.sqrt(np.sum(np.array(list(v1.values())) ** 2))
	return numerator / (v0_len * v1_len)

def create_ppmi_vectors(PMI_func,wids, o_counts, co_counts, tot_count, args):
	'''Creates context vectors for the words in wids, using PPMI.
	These should be sparse vectors.

	:type wids: list of int
	:type o_counts: dict
	:type co_counts: dict of dict
	:type tot_count: int
	:param wids: the ids of the words to make vectors for
	:param o_counts: the counts of each word (indexed by id)
	:param co_counts: the cooccurrence counts of each word pair (indexed by ids)
	:param tot_count: the total number of observations
	:rtype: dict
	:return: the context vectors, indexed by word id
	'''
	vectors = {}
	for wid0 in wids:
		##you will need to change this
		vectors[wid0] = {}
		for widi in co_counts[wid0].keys():
			ppmi = max(PMI_func(co_counts[wid0][widi], o_counts[wid0], o_counts[widi], tot_count, args), 0)  #PPMI
			if ppmi > 0:
				vectors[wid0][widi] = ppmi
	return vectors


def read_counts(filename, wids):
	'''Reads the counts from file. It returns counts for all words, but to
	save memory it only returns cooccurrence counts for the words
	whose ids are listed in wids.

	:type filename: string
	:type wids: list
	:param filename: where to read info from
	:param wids: a list of word ids
	:returns: occurence counts, cooccurence counts, and tot number of observations
	'''
	o_counts = {}  # Occurence counts
	co_counts = {}  # Cooccurence counts
	fp = open(filename)
	N = float(next(fp))
	for line in fp:
		line = line.strip().split("\t")
		wid0 = int(line[0])
		o_counts[wid0] = int(line[1])
		if (wid0 in wids):
			co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
	return (o_counts, co_counts, N)


def print_sorted_pairs(similarities, o_counts, co_counts,first=0, last=100):
	'''Sorts the pairs of words by their similarity scores and prints
	out the sorted list from index first to last, along with the
	counts of each word in each pair.

	:type similarities: dict
	:type o_counts: dict
	:type first: int
	:type last: int
	:param similarities: the word id pairs (keys) with similarity scores (values)
	:param o_counts: the counts of each word id
	:param first: index to start printing from
	:param last: index to stop printing
	:return: none
	'''
	if first < 0: last = len(similarities)
	for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[first:last]:
		word_pair = wid2word[pair[0]]+' vs '+wid2word[pair[1]]

		if pair[1] in co_counts[pair[0]].keys():
			co = co_counts[pair[0]][pair[1]]
		else:
			co = 0
		print("{:.2f}\t{:30}\t{}\t{}\t{}".format(similarities[pair], str(word_pair),
											o_counts[pair[0]], o_counts[pair[1]], co))

def freq_v_sim(sims,path,o_counts):
	xs = []
	ys = []
	for pair in sims.items():
		ys.append(pair[1])
		c0 = o_counts[pair[0][0]]
		c1 = o_counts[pair[0][1]]
		xs.append(min(c0, c1))
	plt.clf()  # clear previous plots (if any)
	plt.xscale('log')  # set x axis to log scale. Must do *before* creating plot
	plt.plot(xs, ys, 'k.')  # create the scatter plot
	plt.xlabel('Min Freq')
	plt.ylabel('Similarity')
	plt.savefig(path+'.png')
	print(path,"Freq vs Similarity Spearman correlation = {:.2f}\n".format(spearmanr(xs, ys)[0]))
	return spearmanr(xs, ys)[0]

def make_pairs(items):
	'''Takes a list of items and creates a list of the unique pairs
	with each pair sorted, so that if (a, b) is a pair, (b, a) is not
	also included. Self-pairs (a, a) are also not included.

	:type items: list
	:param items: the list to pair up
	:return: list of pairs

	'''
	return [(x, y) for x in items for y in items if x < y]


def jaccard_similarity(v0, v1):
	'''Compute the jaccard similarity between two sparse vectors.

		:type v0: dict
		:type v1: dict
		:param v0: first sparse vector
		:param v1: second sparse vector
		:rtype: float
		:return: jaccard between v0 and v1
	'''
	v0_k = set(v0.keys())
	v1_k = set(v1.keys())

	common_keys = v0_k & v1_k
	v0_ = v0_k - common_keys
	v1_ = v1_k - common_keys
	numerator = 0
	denominator = 0
	for k in common_keys:
		numerator += min(v0[k],v1[k])
		denominator += max(v0[k],v1[k])
	for k in v0_:
		denominator += v0[k]
	for k in v1_:
		denominator += v1[k]

	return numerator / denominator

def dice_measure(v0, v1):
	'''Compute the dice measure between two sparse vectors.

		:type v0: dict
		:type v1: dict
		:param v0: first sparse vector
		:param v1: second sparse vector
		:rtype: float
		:return: dice measure between v0 and v1
	'''
	v0_k = set(v0.keys())
	v1_k = set(v1.keys())

	common_keys = v0_k & v1_k
	v0_ = v0_k - common_keys
	v1_ = v1_k - common_keys
	numerator = 0
	denominator = 0
	for k in common_keys:
		numerator += min(v0[k],v1[k])
		denominator += v0[k]+v1[k]
	for k in v0_:
		denominator += v0[k]
	for k in v1_:
		denominator += v1[k]
	return 2 * numerator / denominator

def jsd(v0, v1):
	'''Compute the Jensen–Shannon divergence between two sparse vectors.

		:type v0: dict
		:type v1: dict
		:param v0: first sparse vector
		:param v1: second sparse vector
		:rtype: float
		:return: Jensen–Shannon divergence between v0 and v1
	'''
	M = {}
	for k in v0.keys():
		M[k] = 1/2 * v0[k]
	for k in v1.keys():
		if k in M.keys():
			M[k] = M[k] + 1/2 * v1[k]
		else:
			M[k] = 1/2 * v1[k]
	def D(p,q):
		d = 0
		for k in p.keys():
			d += p[k] * log((p[k] / q[k]))
		return d
	return 0.5 * D(v0,M) + 0.5 *D(v1,M)

def euclidean(v0, v1):
	'''Compute the euclidean distance between two sparse vectors.

		:type v0: dict
		:type v1: dict
		:param v0: first sparse vector
		:param v1: second sparse vector
		:rtype: float
		:return: euclidean distance between v0 and v1
	'''
	common_keys = v0.keys() & v1.keys()
	dist = 0
	dist += sum([(v0[k] - v1[k]) ** 2 for k in common_keys])
	dist += sum([v0[k] ** 2 for k in v0.keys() if k not in common_keys])
	dist += sum([v1[k] ** 2 for k in v1.keys() if k not in common_keys])

	return sqrt(dist)
def cal_similarity(func, vectors, wid_pairs):
	return {(wid0,wid1): func(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

def compare_similarity(all_wids, wid_pairs,o_counts, co_counts, N,V,mark):
	'''
	create PPMI, smoothed PPMI, ttest vectors, calculate cosine similarity, jaccard similarity,
	dice measure, euclidean distance and Jensen-Shannon Divergence
	'''
	#make the word vectors
	vectors = create_ppmi_vectors(PMI,all_wids, o_counts, co_counts, N,None)
	vectors_alpha = create_ppmi_vectors(PMI_alpha, all_wids, o_counts, co_counts,N, args = (None,0.75))
	vectors_t =  create_ppmi_vectors(ttest,all_wids, o_counts, co_counts, N,None)
	#cosine similarity
	c_sims = cal_similarity(cos_sim, vectors, wid_pairs)
	c_sims_alpha =cal_similarity(cos_sim, vectors_alpha, wid_pairs)
	c_sims_t =cal_similarity(cos_sim, vectors_t, wid_pairs)
	#jaccard similarity
	j_sims =cal_similarity(jaccard_similarity, vectors, wid_pairs)
	j_sims_alpha =	cal_similarity(jaccard_similarity, vectors_alpha, wid_pairs)
	j_sims_t=	cal_similarity(jaccard_similarity, vectors_t, wid_pairs)
	#dice measure
	d_sims = cal_similarity(dice_measure, vectors, wid_pairs)
	d_sims_alpha =cal_similarity(dice_measure, vectors_alpha, wid_pairs)
	d_sims_t = cal_similarity(dice_measure, vectors_t, wid_pairs)
	#euclidean distance
	e_sims = cal_similarity(euclidean, vectors, wid_pairs)
	e_sims_alpha = cal_similarity(euclidean, vectors_alpha, wid_pairs)
	e_sims_t = cal_similarity(euclidean, vectors_t, wid_pairs)
	#Jensen-Shannon Divergence
	jsd_sims =cal_similarity(jsd, vectors, wid_pairs)
	jsd_sims_alpha = cal_similarity(jsd, vectors_alpha, wid_pairs)
	jsd_sims_t =  cal_similarity(jsd, vectors_t, wid_pairs)

	sims = {'cosine':c_sims,'cosine_alpha':c_sims_alpha, 'cosine_ttest':c_sims_t,
		'jaccard': j_sims, 'jaccard_alpha':j_sims_alpha, 'jaccard_ttest':j_sims_t,
		'dice':d_sims, 'dice_alpha':d_sims_alpha,'dice_ttest': d_sims_t,
		'ecculidean':e_sims, 'ecculidean_alhpa':e_sims_alpha, 'ecculidean_ttest':e_sims_t,
		'jsd':jsd_sims, 'jsd_alpha':jsd_sims_alpha,'jsd_ttest': jsd_sims_t}

	fvs = {}
	for k in sims.keys():
		print('Sort by %s'%(k))
		sim = sims[k]
		print_sorted_pairs(sim, o_counts,co_counts)
		path =mark+'_'+k
		fvs[k] = freq_v_sim(sim,path,o_counts)
	print('freq vs similarity')
	for k in fvs.keys():

		print('%s\t%.4f'%(k,fvs[k]))

def load_words(path):
	'''
	retreive the word list from a file.
	The file should contains two word pair in each line,
	the words in each line should be separated by '|'
		:type path:string
		:param path: txt file path
		:rtype: set
		:return: the words to calculate similarity
	'''
	lines = open(path).readlines()
	arr = []
	for line in lines:
		arr+= line.strip().split('|')
	return set(arr)
def execute(test_words,mark):
	'''
	execute the whole procedure for vectors creation and similairiy compution
		:type test_words: set
		:type mark:string
		:param test_words: the words to test the similairiy of each other
		:param mark: mark this run is '-t' or '-e'
	'''
	print(test_words)
	stemmed_words = [tw_stemmer(w) for w in test_words]
	all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

	# you could choose to just select some pairs and add them by hand instead
	# but here we automatically create all pairs
	wid_pairs = make_pairs(all_wids)

	#read in the count information (same as in lab)
	(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", all_wids)
	V = len(o_counts.keys())
	compare_similarity(all_wids, wid_pairs, o_counts, co_counts, N,V, mark)
	print('N',N)
	print('sum o_counts values',sum(o_counts.values()))
	print('Vocabulry size, number of keys in o_counts',(len(o_counts.keys())))
'''
-t: test if math formula is correctly programmed and  the similarity of words given by the instructor, "cat", "dog", "mouse", "computer","@justinbieber"
-e path: test the similarity for the words in a given file
'''
if len(sys.argv) <2:

	print("Illegal commands. Try: -t (easy testing); -e words_path (sort by similarity)")
	sys.exit(1)

command = sys.argv[1]
print(command)
if command == '-t':

	'''
	CHECK SIMILARITY TECHNIQUES
	'''
	if (PMI(2, 4, 3, 12) != 1):  # these numbers are from our y,z example
		print("Warning: PMI is incorrectly defined")
	else:
		print("PMI check passed")
	print("PMI check, 2 4 3 12 ",PMI(2, 4, 3, 12))
	print("ttest check, 2 4 3 12 ",ttest(2, 4, 3, 12))
	print('PMI alpha check, 2 4 3 12',PMI_alpha(2, 4, 3, 12,args=(None,0.725)))
	v0 = {1: 1, 2: 1, 3: 2}
	v1 = {2: 2, 4: 1}
	print('cos_sim check:', v0, v1, cos_sim(v0, v1))
	print('jaccard_similarity:', v0, v1, jaccard_similarity(v0, v1))
	print('dice_measure:', v0, v1, dice_measure(v0, v1))
	print('euclidean distance:', v0, v1, euclidean(v0, v1))
	print('Jensen-Shannon Divergence', {1:1.0/10, 2: 9.0/10}, {2:1.0/10,3:9.0/10}, jsd({1:1.0/10, 2: 9.0/10}, {2:1.0/10,3:9.0/10}))
	test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
	print('test words: ', test_words)
	execute(test_words,'check')
	sys.exit(1)
if command == '-e':
	'''
	calculate similarities given the words list
	'''
	words_path = sys.argv[2]
	if not os.path.isfile(words_path):
		print('file path not found',words_path)
		sys.exit(1)
	else:
		test_words = load_words(words_path)
		execute(test_words,words_path[:-4])
		sys.exit(1)
if command not in ['-t','-e']:
	print("Illegal commands. Try: -t (easy testing); -e words_path (sort by similarity)")
	sys.exit(1)
