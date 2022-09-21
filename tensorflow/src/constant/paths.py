from pathlib import Path

# Constants
TF_PATH_KEY = 'tf'
DATASET_PATH_KEY = 'dataset'

# Data: docs
tf_data_dir = Path('../data/tf/')
tf_doc_dir= tf_data_dir / 'r2_7_doc'
tf_api_json_dir = tf_data_dir / 'api_json'
tf_api_text_dir = tf_data_dir / 'api_text'
all_tf_symbol_file = tf_data_dir / 'all_tf_symbols.txt'
standard_tf_symbol_file = tf_data_dir / 'standard_tf_symbols.txt'
func_tf_symbol_file = tf_data_dir / 'func_tf_symbols.txt'
tf_symbol_mapping_file = tf_data_dir / 'tf_symbols_mapping.txt'
all_tf_symbol_url_file = tf_data_dir / 'all_tf_symbols_docurl.txt'
all_tf_symbol_html_file = tf_data_dir / 'all_symbols.html'
all_tf_symbol_url_file = tf_data_dir / 'all_tf_symbols_docurl.txt'
tf_api_match_dir = tf_data_dir / 'match'


# Experiment outputs
expr_dir = Path('../expr')


