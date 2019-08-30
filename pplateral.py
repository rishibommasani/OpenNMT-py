import sys
import csv
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
nltk.download('punkt')
csv.field_size_limit(sys.maxsize)
import pickle
from random import shuffle
import numpy as np

min_size, max_size = 6, 14


def compute_data_statistics(documents, inputs, outputs):
	print("Number of documents: {}".format(len(documents)))
	counts = [len(documents[key]) for key in documents]
	print("Min Sents: {}, Max Sents: {}, Number of sents: {}, Avg. number: {}, Total: {}".format(min(counts), max(counts), sum(counts), sum(counts) / len(counts), len(counts)))
	print("Number of Inputs: {}, Number of Outputs: {}".format(len(inputs), len(outputs)))
	print("An arbitrary input: {}".format(' '.join(inputs[5])))
	print("The corresponding output: {}".format(outputs[5]))
	sc = [len(sent) for sent in inputs]
	print("Statistics for inputs")
	print("Min Words: {}, Max Words: {}, Number of words: {}, Avg. number: {}, Total: {}".format(min(sc), max(sc), sum(sc), sum(sc) / len(sc), len(sc)))
	sc = [len(sent) for sent in outputs]
	print("Statistics for outputs")
	print("Min Words: {}, Max Words: {}, Number of words: {}, Avg. number: {}, Total: {}".format(min(sc), max(sc), sum(sc), sum(sc) / len(sc), len(sc)))
	

def make_data():
	data = {}
	with open('wikipedia_utf8_filtered_20pageviews.csv', mode = 'r') as f:
		for i, row in tqdm(enumerate(f)):
			index, text = row.split(',', 1)
			sent_text = nltk.sent_tokenize(text)
			data[index] = [{'raw_sent' : sent, 'tokenized' : nltk.word_tokenize(sent)} for sent in sent_text]
			if i > 100000:
				break 
	pickle.dump(data, open('wiki.pickle', 'wb'))


def make_files(data, no_stops, noise, partition):
	inputs = data 
	outputs = data 
	stops = set(stopwords.words('english'))
	if no_stops:
		inputs = [[word.lower() for word in sent if word not in stops] for sent in inputs] 
	if noise and noise > 0 and noise <= 1:
		keep_rate = 1 - noise
		inputs = [[word for word in sent if np.random.binomial(1, keep_rate)] for sent in inputs]
	inputs = [sorted(sent) for sent in inputs]
	print(inputs[0], outputs[0])
	combined = list(zip(inputs, outputs))
	shuffle(combined)
	inputs, outputs = zip(*combined)
	# compute_data_statistics(data, inputs, outputs)
	print("Writing outputs")
	with open('{}.wiki.no_stops={}.noise={}.{}.{}-{}_min-max.output.txt'.format(len(data), no_stops, noise, partition, min_size, max_size) , 'w') as f:
		for item in outputs:
			item = ' '.join(item)
			f.write("%s\n" % item)
	print("Writing inputs")
	with open('{}.wiki.no_stops={}.noise={}.{}.{}-{}_min-max.input.txt'.format(len(data), no_stops, noise, partition, min_size, max_size), 'w') as g:
		for item in inputs:
			item = ' '.join(item)
			g.write("%s\n" % item)
	
	#compute_data_statistics(data)


def main():
	pickled = True 
	n = 50000
	test = 5000
	data = pickle.load(open('wiki.pickle', 'rb'))
	data = [sent['tokenized'] for key in data for sent in data[key]]
	data = [sent for sent in data if len(sent) in range(min_size, max_size)]
	start = 0
	data_lengths = {"Train" : 50000, "Valid" : 20000, "Test" : 5000}
	for partition in data_lengths:
		end = start + data_lengths[partition]
		data_slice = data[start : end]
		start = end
		for noise in {None, 0.05, 0.1, 0.2}:
			no_stops = True
			make_files(data_slice, no_stops, noise, partition)
				


	
if __name__ == '__main__':
	main()