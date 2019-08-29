import subprocess

def main():
	train_length = 50000
	valid_length = 50000
	test_length = 2000
	shortened = True
	no_stops = True
	noise = 0.05
	train_src = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=False.input.txt'.format(train_length, shortened, no_stops, noise)
	train_tgt = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=False.output.txt'.format(train_length, shortened, no_stops, noise)
	valid_src = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=True.input.txt'.format(valid_length, shortened, no_stops, noise)
	valid_tgt = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=True.output.txt'.format(valid_length, shortened, no_stops, noise)
	save_data = 'data/{}.wiki.shortened={}.no_stops={}.noise={}'.format(train_length, shortened, no_stops, noise)
	model = 'models/{}.wiki.shortened={}.no_stops={}.noise={}'.format(train_length, shortened, no_stops, noise)
	checkpoint = 40000
	checkpoint_file = 'models/{}.wiki.shortened={}.no_stops={}.noise={}_step_{}.pt'.format(train_length, shortened, no_stops, noise, checkpoint)
	#test_src = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=Test.input.txt'.format(test_length, shortened, no_stops, noise)
	#test_system_out = 'output/{}.wiki.shortened={}.no_stops={}.noise={}.valid=Test.system.output.txt'.format(test_length, shortened, no_stops, noise)
	test_src = 'sotu_0.700_10k_doclength10_topics_200epochs.txt'
	test_system_out = 'sotu_0.700_10k_doclength10_topics_200epochs.gen.txt'
	# for noise in {None, 0.05, 0.1, 0.2}:
	# 	for no_stops in {True, False}:
	# 		train_src = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=False.input.txt'.format(train_length, shortened, no_stops, noise)
	# 		train_tgt = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=False.output.txt'.format(train_length, shortened, no_stops, noise)
	# 		valid_src = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=True.input.txt'.format(valid_length, shortened, no_stops, noise)
	# 		valid_tgt = '{}.wiki.shortened={}.no_stops={}.noise={}.valid=True.output.txt'.format(valid_length, shortened, no_stops, noise)
	# 		save_data = 'data/{}.wiki.shortened={}.no_stops={}.noise={}'.format(train_length, shortened, no_stops, noise)
	# 		model = 'models/{}.wiki.shortened={}.no_stops={}.noise={}'.format(train_length, shortened, no_stops, noise)
	# preprocess = ['python3', 'preprocess.py', '-train_src', train_src, '-train_tgt', train_tgt, '-valid_src', valid_src, '-valid_tgt', valid_tgt, '-save_data', save_data, '-src_seq_length', '10000', '-tgt_seq_length', '10000', '-src_seq_length_trunc', '100', '-tgt_seq_length_trunc', '100', '-dynamic_dict', '-share_vocab', '-shard_size', '10000']	
	# subprocess.check_call(preprocess)
	# train = ['python3', 'train.py', '-save_model', model, '-data', save_data, '-copy_attn', '-global_attention', 'mlp', '-word_vec_size', '128', '-rnn_size', '512', '-layers', '1', '-encoder_type', 'brnn', '-train_steps', '200000', '-max_grad_norm', '2', '-dropout', '0.', '-batch_size', '16', '-valid_batch_size', '16', '-optim', 'adagrad', '-learning_rate', '0.15', '-adagrad_accumulator_init', '0.1', '-reuse_copy_attn', '-copy_loss_by_seqlength', '-bridge', '-seed', '777', '-copy_attn_force', '-coverage_attn', '-lambda_coverage', '1', '-save_checkpoint_steps', '5000', '-valid_steps', '10000']
	# subprocess.check_call(train)
	# example_test = "example_test.txt"
	test = ['python3', 'translate.py', '-batch_size', '40', '-beam_size', '10', '-model', checkpoint_file, '-src', test_src, '-output', test_system_out, '-min_length', '5', '-verbose', '-stepwise_penalty', '-coverage_penalty', 'summary', '-beta', '5', '-length_penalty', 'wu', '-alpha', '0.9', '-verbose', '-block_ngram_repeat', '3']
	subprocess.check_call(test)

if __name__ == '__main__':
	main()