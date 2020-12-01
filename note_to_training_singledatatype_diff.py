import argparse
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.lm import NgramCounter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import parmap
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
    parser = argparse.ArgumentParser(description='Preprocess notes, extract ventilation duration, and prepare for running MixEHR. Differentiating between different data types.')
    parser.add_argument('input_file_path', help='path to the original file')
    parser.add_argument('output_directory', help='directory to store outputs')
    parser.add_argument('--no_data', action='store_true', help='NOT outputing data')
    parser.add_argument('--use_vocab', help='specify directory containing the vocabulary to use. setting no_vocab to true')
    parser.add_argument('--no_vocab', action='store_true', help='NOT storing vocabulary')
    parser.add_argument('-m', '--meta_data', action='store_true', help='output meta data (default not)')
    parser.add_argument('-v', '--ventilation_duration', action='store_true', help='output ventilation duration (default not)')
    parser.add_argument('--binarized_mv', action='store_true', help='using Label in files that do not have continuous MV durations')
    parser.add_argument('-s', '--split_datasets', action='store_true', help='split the whole cohort into train/validation/test (default not)')
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
    if args.ventilation_duration:
        if not args.binarized_mv:
            print('Outputting continuous MV duration')
        else:
            print('Outputting binarized MV duration')
    if args.split_datasets:
        print('Splitting into train/valiation/test')
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

def merge_notes(hadm_id, category):
    tmp_data = data[data.HADM_ID == hadm_id]
    return "\n".join(tmp_data[tmp_data.CATEGORY == category].TEXT)

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
    ventilation_duration = []
    for idx, id in tqdm(enumerate(ids)):
        set_ventilation = False # indicate whether ventilation duration is calculated for current HADM_ID
        if not args.no_data or args.ventilation_duration:
            for _, row in data[data['HADM_ID'] == id].iterrows(): # for one HADM_ID - CATEGORY pair
                unigram_counts = get_unigram_counts(row["PROCTEXT"])
                for word in set(row["PROCTEXT"].split()):
                    if word not in word_indexes[row['CATEGORY']].keys():
                        continue
                    if not args.no_data:
                        output.append(" ".join([str(idx), category_to_id[row['CATEGORY']], str(word_indexes[row['CATEGORY']][word]), "0", str(unigram_counts[word])]))
                    if args.ventilation_duration and not set_ventilation:
                        set_ventilation = True
                        if not args.binarized_mv:
                            ventilation_duration.append(" ".join([str(idx), str(data[data.HADM_ID == id]["DURATION"].values[0])]))
                        else:
                            ventilation_duration.append(" ".join([str(idx), str(data[data.HADM_ID == id]["BI_DURATION"].values[0])]))
    
#     for idx, id in tqdm(enumerate(ids)):
#         set_ventilation = False # indicate whether ventilation duration is calculated
#         if not args.no_data or args.ventilation_duration:
#             unigram_counts = get_unigram_counts(data[data.HADM_ID == id]["PROCTEXT"].tolist())
#             for word in set(data[data.HADM_ID == id]["PROCTEXT"].values[0].split()):
#                 if word not in word_index.keys():
#                     continue
#                 if not args.no_data:
#                     output.append(" ".join([str(idx), "1", str(word_index[word]), "0", str(unigram_counts[word])]))
#                 if args.ventilation_duration and not set_ventilation:
#                     set_ventilation = True
#                     ventilation_duration.append(" ".join([str(idx), str(data[data.HADM_ID == id]["DURATION"].values[0])]))
    return output, ventilation_duration

if __name__ == '__main__':
    args = parse_args()
    if args.no_vocab and args.no_data and not args.meta_data and not args.ventilation_duration:
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
    merged_data = data[data['CATEGORY'].isin(categories)][["HADM_ID", "CATEGORY"]].drop_duplicates()
    if args.discard_ids:
        discard_ids_df = pd.read_csv(args.discard_ids, header=None)
        discard_ids_df.columns = ['HADM_ID']
        discard_ids = discard_ids_df['HADM_ID'].tolist()
        merged_data = merged_data[~merged_data['HADM_ID'].isin(discard_ids)]
        print('[' + time.ctime() + ']', 'discarding', len(discard_ids), 'HADM_IDs')
    with Pool(processes=args.max_procs) as pool:
        merged_data["ALLTEXT"] = pool.starmap(merge_notes, zip(merged_data['HADM_ID'], merged_data['CATEGORY']))
    merged_data.dropna()
    print('[' + time.ctime() + ']', 'after merging notes of same HADM_ID and same CATEGORY, we have', merged_data.shape[0], 'unique HADM_ID - CATEGORY pairs')

    # preprocess notes
    with Pool(processes=args.max_procs) as pool:
        merged_data["PROCTEXT"] = pool.map(preprocess_text, merged_data['ALLTEXT'])
    print('[' + time.ctime() + ']', "notes preprocessed")

    # get ventilation duration
    if args.ventilation_duration:
        if not args.binarized_mv:
            try:
                merged_data['STARTTIME'] = pd.to_datetime(merged_data.HADM_ID.apply(lambda hadm_id: np.min(data[data.HADM_ID == hadm_id].STARTTIME)))
            except:
                # accommadating files that use FIRST_VENT_STARTTIME to represent STARTTIME
                merged_data['STARTTIME'] = pd.to_datetime(merged_data.HADM_ID.apply(lambda hadm_id: np.min(data[data.HADM_ID == hadm_id].FIRST_VENT_STARTTIME)))
            merged_data['ENDTIME'] = pd.to_datetime(merged_data.HADM_ID.apply(lambda hadm_id: np.max(data[data.HADM_ID == hadm_id].ENDTIME)))
            merged_data['DURATION'] = pd.to_timedelta(merged_data['ENDTIME'] - merged_data['STARTTIME'], unit='h') / np.timedelta64(1, 'h')
            merged_data.drop(columns=['STARTTIME', 'ENDTIME'])
            print('[' + time.ctime() + ']', "ventilation duration calculated")
        else:
            # accommadating files that have no information of MV sessions but only labels indicating prolonged or not
            print('[' + time.ctime() + ']', "WARNING: no continuous MV duration available, using Label")
            merged_data['BI_DURATION'] = merged_data.HADM_ID.apply(lambda hadm_id: data[data.HADM_ID == hadm_id].Label.values[0])
            print('[' + time.ctime() + ']', "binarized ventilation duration calculated")

    # split into train/valid/test sets
    if args.split_datasets:
        train_data = merged_data.sample(frac=0.6, random_state=1)
        valid_data = merged_data.drop(train_data.index).sample(frac=0.5, random_state=1)
        test_data = merged_data.drop(train_data.index).drop(valid_data.index)
        held_out_ids = pd.concat([valid_data['HADM_ID'], test_data['HADM_ID']], ignore_index=True)
        train_data.to_csv(os.path.join(args.output_directory, 'train_notes.csv'), header=False, index=False)   
        valid_data.to_csv(os.path.join(args.output_directory, 'validation_notes.csv'), header=False, index=False) 
        test_data.to_csv(os.path.join(args.output_directory, 'test_notes.csv'), header=False, index=False) 
        held_out_ids.to_csv(os.path.join(args.output_directory, 'held_out_ids.csv'), header=False, index=False)
        print('[' + time.ctime() + ']', 'train/valid/test sets written to', args.output_directory)
        
    # generate word indexes
    if not args.use_vocab:
        if args.split_datasets:
            word_indexes = {category: get_word_index(train_data[train_data['CATEGORY'] == category]) for category in categories}
        else:
            word_indexes = {category: get_word_index(merged_data[merged_data['CATEGORY'] == category]) for category in categories}
        print('[' + time.ctime() + ']', 'word indexes generated')
    else:
        word_indexes = {}
        for category in categories:
            with open(os.path.join(args.use_vocab, category_to_id[category] + '_vocab.txt'), 'r') as file:
                vocab = [line.rstrip('\n') for line in file.readlines()]
            word_indexes[category] = {line.split(',')[0]: line.split(',')[1] for line in vocab}
        print('[' + time.ctime() + ']', 'read word indexes from', args.use_vocab)

    # store vocabulary
    if not args.no_vocab:
        for category in categories:
            word_index_output = [','.join([word, str(idx)]) for word, idx in word_indexes[category].items()]
            with open(os.path.join(args.output_directory, category_to_id[category] + '_vocab.txt'), "w") as file:
                file.writelines('\n'.join(word_index_output))
        print('[' + time.ctime() + ']', 'vocabulary written to', args.output_directory)
            
    # create output dataframe
    if not args.split_datasets:
        output, ventilation_duration = create_output_dataframe(merged_data, word_indexes, args)
    else:
        with Pool(processes=args.max_procs) as pool:
            (train_output, train_ventilation_duration), (valid_output, valid_ventilation_duration), (test_output, test_ventilation_duration) = \
            pool.starmap(create_output_dataframe, zip([train_data, valid_data, test_data], repeat(word_indexes), repeat(args)))
        
    print('[' + time.ctime() + ']', "data ready")
    
    # create meta data
    if args.meta_data:
        meta_data = []
        for category in categories:
            meta_data += [" ".join([category_to_id[category], str(idx), "1"]) for idx in word_indexes[category].values()]
        print('[' + time.ctime() + ']', "meta data ready")
    
    # write to output
    if not args.no_data:
        if not args.split_datasets:
            with open(os.path.join(args.output_directory, 'data.txt'), "w") as file:
                file.writelines("\n".join(output))
            print('[' + time.ctime() + ']', "data written to", os.path.join(args.output_directory, 'data.txt'))
        else:
            with open(os.path.join(args.output_directory, 'train_data.txt'), "w") as file:
                file.writelines("\n".join(train_output))
            with open(os.path.join(args.output_directory, 'validation_data.txt'), "w") as file:
                file.writelines("\n".join(valid_output))
            with open(os.path.join(args.output_directory, 'test_data.txt'), "w") as file:
                file.writelines("\n".join(test_output))
            print('[' + time.ctime() + ']', "data written to", os.path.join(args.output_directory, 'train_data/validation_data/test_data.txt'))
    if args.meta_data:
        with open(os.path.join(args.output_directory, 'meta.txt'), "w") as file:
            file.writelines("\n".join(meta_data))
        print('[' + time.ctime() + ']', "meta data written to", os.path.join(args.output_directory, 'meta.txt'))
    if args.ventilation_duration:
        if not args.split_datasets:
            if not args.binarized_mv:
                with open(os.path.join(args.output_directory, 'vent.txt'), "w") as file:
                    file.writelines("\n".join(ventilation_duration))
                print('[' + time.ctime() + ']', "ventilation duration written to", os.path.join(args.output_directory, 'vent.txt'))
            else:
                with open(os.path.join(args.output_directory, 'bi_vent.txt'), "w") as file:
                    file.writelines("\n".join(ventilation_duration))
                print('[' + time.ctime() + ']', "binarized ventilation duration written to", os.path.join(args.output_directory, 'bi_vent.txt'))
        else:
            if not args.binarized_mv:
                with open(os.path.join(args.output_directory, 'train_vent.txt'), "w") as file:
                    file.writelines("\n".join(train_ventilation_duration))
                with open(os.path.join(args.output_directory, 'validation_vent.txt'), "w") as file:
                    file.writelines("\n".join(valid_ventilation_duration))
                with open(os.path.join(args.output_directory, 'test_vent.txt'), "w") as file:
                    file.writelines("\n".join(test_ventilation_duration))
                print('[' + time.ctime() + ']', "ventilation duration written to", os.path.join(args.output_directory, 'train_vent/validation_vent/test_vent.txt'))
            else:
                with open(os.path.join(args.output_directory, 'train_bi_vent.txt'), "w") as file:
                    file.writelines("\n".join(train_ventilation_duration))
                with open(os.path.join(args.output_directory, 'validation_bi_vent.txt'), "w") as file:
                    file.writelines("\n".join(valid_ventilation_duration))
                with open(os.path.join(args.output_directory, 'test_bi_vent.txt'), "w") as file:
                    file.writelines("\n".join(test_ventilation_duration))
                print('[' + time.ctime() + ']', "binarized ventilation duration written to", os.path.join(args.output_directory, 'train_bi_vent/validation_bi_vent/test_bi_vent.txt'))
