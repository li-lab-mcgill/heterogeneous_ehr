import argparse
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.lm import NgramCounter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, pickle
from multiprocessing import Pool
from itertools import repeat
import time

# categories and type ids
category_to_id = {
    'Physician ': '1',
    'Nursing': '2',
    'Respiratory ': '3',
    'Nursing/other': '4',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess notes and prepare for running MixEHR. Not differentiating between different data types.')
    parser.add_argument('input_file_path', help='path to the original file')
    parser.add_argument('output_directory', help='directory to store outputs')
    parser.add_argument('--no_data', action='store_true', help='NOT outputing data')
    parser.add_argument('--use_vocab', help='specify directory containing the vocabulary to use. setting no_vocab to true')
    parser.add_argument('--no_vocab', action='store_true', help='NOT storing vocabulary')
    parser.add_argument('-m', '--meta_data', action='store_true', help='output meta data (default not)')
    parser.add_argument('--discard_ids', help='csv file containing the HADM_IDs that should be discarded')
    parser.add_argument('--no_phy', action='store_true', help='NOT using physicians notes')
    parser.add_argument('--no_nur', action='store_true', help='NOT using nurses notes')
    parser.add_argument('--res', action='store_true', help='using respiratory notes')
    parser.add_argument('--other', action='store_true', help='using nursing/other notes. WARNING: this category contain other types of notes e.g. discharge summary, use with caution')
    parser.add_argument('--max_procs', type=int, help='maximal number of processes (default 10)', default=10)
    return parser.parse_args()

# print args for double check
def print_args(args):
    print('Input file:', args.input_file_path)
    print('Output directory:', args.output_directory)
    print('Using', args.max_procs, 'cores')
    if args.no_data:
        print('Not outputting data file for mixehr')
    if args.use_vocab:
        print('Using vocabulary from', args.use_vocab)
    if args.no_vocab:
        print('Not outputting vocabulary')
    if args.meta_data:
        print('Outputting meta data')
    if args.discard_ids:
        print('Discarding HADM_IDs in', args.discard_ids)
    if args.no_phy:
        print('Not using physician notes')
    if args.no_nur:
        print('Not using nursing notes')
    if args.res:
        print('Using respiratory notes')
    if args.other:
        print('Using nursing/other notes')

def merge_notes(hadm_id):
    return "\n".join(data[data.HADM_ID == hadm_id].TEXT)

def preprocess_text(text):
    # convert to lower case
    lower = text.lower()
    # remove punc
    no_punc = lower.translate(str.maketrans("", "", string.punctuation))
    # remove white space
    no_white = " ".join(no_punc.split())
    # remove numbers
    no_num = no_white.translate(str.maketrans("", "", string.digits))
    # remove stopwords
    normal_sws = list(set(stopwords.words('english')))
#     more_sws = pickle.load(open('/home/mcb/li_lab/zwen8/data/mimic/more_stopwords.pickle', 'rb'))
#     sws = set(normal_sws + more_sws)
    no_sw = " ".join([word for word in no_num.split() if word not in normal_sws])
    return no_sw

def get_unigram_counts(text):
#     text_word = [sentence.split() for sentence in text]
    text_word = [text.split()]
    text_unigram = [ngrams(sentence, 1) for sentence in text_word]
    return NgramCounter(text_unigram)

def get_word_index(data, max_df=0.15, min_df=5):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    vectorizer.fit(data['PROCTEXT'])
    return {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}

def create_output_dataframe(data, word_indexes, args):
    ids = data["HADM_ID"].unique().tolist()  # HADM_IDs
    output = []
    mortality_label = []
    for id in tqdm(ids):
        set_label = False # indicate whether label is recorded for current HADM_ID
        if not args.no_data:
            for _, row in data[data['HADM_ID'] == id].iterrows(): # for one HADM_ID
                unigram_counts = get_unigram_counts(row["PROCTEXT"])
                for word in set(row["PROCTEXT"].split()):
                    if word not in word_index.keys():
                        continue
                    if not args.no_data:
                        output.append(" ".join([str(int(id)), "1", str(word_index[word]), "0", str(unigram_counts[word])]))
                    if not set_label:
                        set_label = True
                        mortality_label.append(" ".join([str(id), str(data[data.HADM_ID == id]["MORTALITY_LABEL"].values[0])]))

    return output, mortality_label

if __name__ == '__main__':
    args = parse_args()
    if args.no_vocab and args.no_data and not args.meta_data:
        raise Exception('no output specified')
    if args.no_phy and args.no_nur and not args.res:
        raise Exception('not using any type of note (physicians, nurses, respiratory, nursing/other)')
    if args.use_vocab:
        args.no_vocab = True
        
    # print arguments for double check
    print_args(args)

    data = pd.read_csv(args.input_file_path)
    print('[' + time.ctime() + ']', "data read")

    categories = []
    if not args.no_phy:
        categories.append('Physician ')
    if not args.no_nur:
        categories.append('Nursing')
    if args.res:
        categories.append('Respiratory ')
    if args.other:
        categories.append('Nursing/other')
    args.categories = categories
    
    # merge notes of same HADM_ID of specified categories
    merged_data = data[data['CATEGORY'].isin(categories)][["HADM_ID", "MORTALITY_LABEL"]].drop_duplicates()
    if args.discard_ids:
        discard_ids_df = pd.read_csv(args.discard_ids, header=None)
        discard_ids_df.columns = ['HADM_ID']
        discard_ids = discard_ids_df['HADM_ID'].tolist()
        merged_data = merged_data[~merged_data['HADM_ID'].isin(discard_ids)]
        print('[' + time.ctime() + ']', 'discarding', len(discard_ids), 'HADM_IDs')
    with Pool(processes=args.max_procs) as pool:
        merged_data["ALLTEXT"] = pool.map(merge_notes, merged_data['HADM_ID'])
    merged_data.dropna()
    print('[' + time.ctime() + ']', 'after merging notes of same HADM_ID, we have', merged_data.shape[0], 'unique HADM_ID pairs')

    # preprocess notes
    with Pool(processes=args.max_procs) as pool:
        merged_data["PROCTEXT"] = pool.map(preprocess_text, merged_data['ALLTEXT'])
    print('[' + time.ctime() + ']', "notes preprocessed")
        
    # generate word indexes
    if not args.use_vocab:
        word_index = get_word_index(merged_data)
        print('[' + time.ctime() + ']', 'word index generated')
    else:
        with open(os.path.join(args.use_vocab, 'vocab.txt'), 'r') as file:
            vocab = [line.rstrip('\n') for line in file.readlines()]
        word_index = {line.split(',')[0]: line.split(',')[1] for line in vocab}
        print('[' + time.ctime() + ']', 'read word index from', args.use_vocab)

    # store vocabulary
    if not args.no_vocab:
        word_index_output = [','.join([word, str(idx)]) for word, idx in word_index.items()]
        with open(os.path.join(args.output_directory, 'vocab.txt'), "w") as file:
            file.writelines('\n'.join(word_index_output))
        print('[' + time.ctime() + ']', 'vocabulary written to', args.output_directory)
            
    # create output dataframe
    output, mortality_label = create_output_dataframe(merged_data, word_index, args)
        
    print('[' + time.ctime() + ']', "data ready")
    
    # create meta data
    if args.meta_data:
        meta_data = [" ".join(["1", str(idx), "1"]) for idx in word_index.values()]
        print('[' + time.ctime() + ']', "meta data ready")
    
    # write to output
    if not args.no_data:
        with open(os.path.join(args.output_directory, 'data.txt'), "w") as file:
            file.writelines("\n".join(output))
        print('[' + time.ctime() + ']', "data written to", os.path.join(args.output_directory, 'data.txt'))
    if args.meta_data:
        with open(os.path.join(args.output_directory, 'meta.txt'), "w") as file:
            file.writelines("\n".join(meta_data))
        print('[' + time.ctime() + ']', "meta data written to", os.path.join(args.output_directory, 'meta.txt'))

    with open(os.path.join(args.output_directory, 'label.txt'), "w") as file:
        file.writelines("\n".join(mortality_label))
    print('[' + time.ctime() + ']', "mortality label written to", os.path.join(args.output_directory, 'label.txt'))
