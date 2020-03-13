import argparse
import pandas as pd
import numpy as np

def get_arguments(input_array = []):
	parser = argparse.ArgumentParser(description='Fetch confidence score report')
	parser.add_argument('--almond', type=str, default=None, help="In-distribution (Almond) test file results")
	parser.add_argument('--mnli', type=str, default=None, help="Out-of-distribution (MNLI) test file results")
	parser.add_argument('--wiki', type=str, default=None, help="Out-of-distribution (WIKI) test file results")
	parser.add_argument('--dir', type=str, default='./', help="directory to output results")
	if not input_array:
		args = parser.parse_args()
	else:
		args = parser.parse_args(input_array)
	return args

def convert_result_to_df(lst, label):
	return pd.DataFrame(list(zip(lst, [label]*len(lst))), columns=['prediction', 'label'])

def fetch_results(file_obj):
	results = []
	for line in file_obj:
		tensor_str = line.split()[1]
		begin_str = "tensor("	
		end_str = ", device"
		try:
			begin_index = tensor_str.index(begin_str)
			end_index = -1 # tensor_str.index(end_str)
			float_str = tensor_str[begin_index+len(begin_str) : end_index]
			results.append(float(float_str))
		except:
			continue
	return results

def interpret_results(result):
	return [np.sum(result.prediction == 0), np.sum(result.prediction == 1), np.sum(result.prediction == result.label)]	

def main():
	# ['--almond', 'eval_dir_l01_min/valid/almond.tsv', '--mnli', '', '--wiki', '']
	args = get_arguments()
	almond_file = open(args.almond, 'r') if args.almond else None
	mnli_file = open(args.mnli, 'r') if args.mnli else None
	wiki_file = open(args.wiki, 'r') if args.wiki else None

	almond_result, mnli_result, wiki_result = None, None, None

	# fetch almond results
	if almond_file:
		almond_result = fetch_results(almond_file)
		almond_result = convert_result_to_df(almond_result, 0)

	return almond_result, mnli_result, wiki_result, args.dir

if __name__ == '__main__':
	almond_result, mnli_result, wiki_result, output_dir = main()
	
	return_val = []
	indices = []
	if almond_result is not None:	
		return_val.append(interpret_results(almond_result))
		indices.append("Almond")
	if mnli_result is not None:
		return_val.append(interpret_results(mnli_result))
		indices.append("MNLI")
	if wiki_result is not None:
		return_val.append(interpret_results(wiki_result))
		indices.append("WIKI")
	print(pd.DataFrame(return_val, index=indices, columns=["0", "1", "Correct"]))

	# if almond_result is not None:
	# 	almond_result.to_csv(output_dir + "almond.csv", index=False)
	# if mnli_result is not None:
	# 	mnli_result.to_csv(output_dir + "mnli.csv", index=False)
	# if wiki_result is not None:
	# 	wiki_result.to_csv(output_dir + "wiki.csv", index=False)
