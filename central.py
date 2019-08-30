import subprocess

def main():
	train_length = 50000
	valid_length = 20000
	test_length = 5000
	no_stops = True
	min_length = 6
	max_length = 14
	for noise in [None, 0.05, 0.1, 0.2]:
		train_src = '{}.wiki.no_stops={}.noise={}.Train.{}-{}_min-max.input.txt'.format(train_length,  no_stops, noise, min_length, max_length)
		train_tgt = '{}.wiki.no_stops={}.noise={}.Train.{}-{}_min-max.output.txt'.format(train_length,  no_stops, noise, min_length, max_length)

		valid_src = '{}.wiki.no_stops={}.noise={}.Valid.{}-{}_min-max.input.txt'.format(valid_length,  no_stops, noise, min_length, max_length)
		valid_tgt = '{}.wiki.no_stops={}.noise={}.Valid.{}-{}_min-max.output.txt'.format(valid_length, no_stops, noise, min_length, max_length)

		save_data = 'data/{}.wiki.no_stops={}.noise={}.{}-{}_min-max.'.format(train_length, no_stops, noise, min_length, max_length)
		model = 'models/{}.minmaxsize={}-{}.wiki.no_stops={}.noise={}'.format(train_length, min_length, max_length, no_stops, noise)
		checkpoint = 80000
		checkpoint_file = 'models/{}.minmaxsize={}-{}.wiki.no_stops={}.noise={}_step_{}.pt'.format(train_length, min_length, max_length, no_stops, noise, checkpoint)
		
		test_src = '{}.wiki.no_stops={}.noise={}.Test.{}-{}_min-max.input.txt'.format(test_length, no_stops, noise, min_length, max_length)
		test_system_out = 'output/{}.wiki.no_stops={}.noise={}.Test.{}-{}_min-max.{}_steps.system.output.txt'.format(test_length, no_stops, noise, min_length, max_length, checkpoint)
		
		# preprocess = ['python3', 'preprocess.py', '-train_src', train_src, '-train_tgt', train_tgt, '-valid_src', valid_src, '-valid_tgt', valid_tgt, '-save_data', save_data, '-src_seq_length', '10000', '-tgt_seq_length', '10000', '-src_seq_length_trunc', '100', '-tgt_seq_length_trunc', '100', '-dynamic_dict', '-share_vocab', '-shard_size', '10000']	
		# subprocess.check_call(preprocess)
		
		# train = ['python3', 'train.py', '-save_model', model, '-data', save_data, '-copy_attn', '-global_attention', 'mlp', '-word_vec_size', '128', '-rnn_size', '512', '-layers', '1', '-encoder_type', 'brnn', '-train_steps', '100000', '-max_grad_norm', '2', '-dropout', '0.', '-batch_size', '16', '-valid_batch_size', '16', '-optim', 'adagrad', '-learning_rate', '0.15', '-adagrad_accumulator_init', '0.1', '-reuse_copy_attn', '-copy_loss_by_seqlength', '-bridge', '-seed', '777', '-copy_attn_force', '-coverage_attn', '-lambda_coverage', '1', '-save_checkpoint_steps', '20000', '-valid_steps', '20000']
		# subprocess.check_call(train)
		
		test = ['python3', 'translate.py', '-verbose', '-batch_size', '40', '-beam_size', '10', '-model', checkpoint_file, '-src', test_src, '-output', test_system_out, '-min_length', '5', '-stepwise_penalty', '-coverage_penalty', 'summary', '-beta', '5', '-length_penalty', 'wu', '-alpha', '0.9', '-block_ngram_repeat', '3']
		subprocess.check_call(test)

if __name__ == '__main__':
	main()