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

# categories and type ids
category_to_id = {
    'Physician ': '1',
    'Nursing': '2',
    'Respiratory ': '3',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess notes, extract ventilation duration, and prepare for running MixEHR')
    parser.add_argument('input_file_path', help='path to the original file')
    parser.add_argument('output_directory', help='directory to store outputs')
    parser.add_argument('--no_data', action='store_true', help='NOT outputing data')
    parser.add_argument('--use_vocab', help='specify vocabulary to use. setting no_vocab to true')
    parser.add_argument('--no_vocab', action='store_true', help='NOT storing vocabulary')
    parser.add_argument('-m', '--meta_data', action='store_true', help='output meta data (default not)')
    parser.add_argument('-v', '--ventilation_duration', action='store_true', help='output ventilation duration (default not)')
    parser.add_argument('-s', '--split_datasets', action='store_true', help='split the whole cohort into train/validation/test (default not)')
    parser.add_argument('--no_phy', action='store_true', help='NOT using physicians notes')
    parser.add_argument('--no_nur', action='store_true', help='NOT using nurses notes')
    parser.add_argument('--res', action='store_true', help='using respiratory notes')
    parser.add_argument('--max_procs', type=int, help='maximal number of processes (default 10)', default=10)
    return parser.parse_args()

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
    text_word = [sentence.split() for sentence in text]
    text_unigram = [ngrams(sentence, 1) for sentence in text_word]
    return NgramCounter(text_unigram)

def create_output_dataframe(data, word_index, args):
    ids = data["HADM_ID"].unique().tolist()  # HADM_IDs
    output = []
    ventilation_duration = []
    for idx, id in tqdm(enumerate(ids)):
        set_ventilation = False # indicate whether ventilation duration is calculated
        if not args.no_data or args.ventilation_duration:
            for _, row in data[data['HADM_ID'] == id].iterrows():
                unigram_counts = get_unigram_counts(row["PROCTEXT"].tolist())
                for word in set(row["PROCTEXT"].values[0].split()):
                    if word not in word_index.keys():
                        continue
                    if not args.no_data:
                        output.append(" ".join([str(idx), category_to_id[row['CATEGORY']], str(word_index[word]), "0", str(unigram_counts[word])]))
                    if args.ventilation_duration and not set_ventilation:
                        set_ventilation = True
                        ventilation_duration.append(" ".join([str(idx), str(data[data.HADM_ID == id]["DURATION"].values[0])]))
    
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
        raise Exception('not using any type of note (physicians, nurses, respiratory)')
    if args.use_vocab:
        args.no_vocab = True

    data = pd.read_csv(args.input_file_path)
    print("data read")

    categories = []
    if not args.no_phy:
        categories.append('Physician ')
    if not args.no_nur:
        categories.append('Nursing')
    if args.res:
        categories.append('Respiratory ')
    args.categories = categories
    
    # merge notes of same HADM_ID of specified categories
    merged_data = data[data['CATEGORY'].isin(categories)][['SUBJECT_ID', "HADM_ID", "CATEGORY"]].drop_duplicates()
#     merged_data["ALLTEXT"] = merged_data.HADM_ID.apply(lambda hadm_id: "\n".join(data[data.HADM_ID == hadm_id].TEXT))
    with Pool(processes=args.max_procs) as pool:
        merged_data["ALLTEXT"] = pool.starmap(merge_notes, zip(merged_data['HADM_ID'], merged_data['CATEGORY']))
#     parmap.starmap(merge_notes, list(zip(merged_data['HADM_ID'], merged_data['CATEGORY'])), pm_processes=10, pm_pbar=True)
    merged_data.dropna()
    print('after merging notes of same HADM_ID and same CATEGORY, we have', merged_data.shape[0], 'unique HADM_ID - CATEGORY pairs')

    # preprocess notes
#     merged_data["PROCTEXT"] = merged_data.ALLTEXT.apply(lambda text: preprocess_text(text))
    with Pool(processes=args.max_procs) as pool:
        merged_data["PROCTEXT"] = pool.map(preprocess_text, merged_data['ALLTEXT'])
    print("notes preprocessed")

    # get ventilation duration
    if args.ventilation_duration:
        merged_data['STARTTIME'] = pd.to_datetime(merged_data.HADM_ID.apply(lambda hadm_id: np.min(data[data.HADM_ID == hadm_id].STARTTIME)))
        merged_data['ENDTIME'] = pd.to_datetime(merged_data.HADM_ID.apply(lambda hadm_id: np.max(data[data.HADM_ID == hadm_id].ENDTIME)))
        merged_data['DURATION'] = pd.to_timedelta(merged_data['ENDTIME'] - merged_data['STARTTIME'], unit='h') / np.timedelta64(1, 'h')
        merged_data.drop(columns=['STARTTIME', 'ENDTIME'])
        print("ventilation duration calculated")

    # split into train/valid/test sets
    if args.split_datasets:
        train_data = merged_data.sample(frac=0.8, random_state=1)
        valid_data = merged_data.drop(train_data.index).sample(frac=0.5, random_state=1)
        test_data = merged_data.drop(train_data.index).drop(valid_data.index)
        train_data.to_csv(os.path.join(args.output_directory, 'train_notes.txt'), header=False, index=False)   
        valid_data.to_csv(os.path.join(args.output_directory, 'validation_notes.txt'), header=False, index=False) 
        test_data.to_csv(os.path.join(args.output_directory, 'test_notes.txt'), header=False, index=False) 
        print('train/valid/test sets written to', args.output_directory)
        
    # generate word indexes
    if not args.use_vocab:
        vectorizer = CountVectorizer(max_df=0.1, min_df=5)
        vectorizer.fit(merged_data['PROCTEXT'])
        word_index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
        print('word indexes generated')
    else:
        with open(args.use_vocab, 'r') as file:
            vocab = [line.rstrip('\n') for line in file.readlines()]
        word_index = {line.split(',')[0]: line.split(',')[1] for line in vocab}
        print('read word indexes from', args.use_vocab)

    # store vocabulary
    if not args.no_vocab:
        word_index_output = [','.join([word, str(idx)]) for word, idx in word_index.items()]
        with open(os.path.join(args.output_directory, 'vocab.txt'), "w") as file:
            file.writelines('\n'.join(word_index_output))
        print('vocabulary written to', os.path.join(args.output_directory, 'vocab.txt'))
            
    # create output dataframe
    if not args.split_datasets:
        output, ventilation_duration = create_output_dataframe(merged_data, word_index, args)
    else:
        with Pool(processes=args.max_procs) as pool:
            (train_output, train_ventilation_duration), (valid_output, valid_ventilation_duration), (test_output, test_ventilation_duration) = \
            pool.starmap(create_output_dataframe, zip([train_data, valid_data, test_data], repeat(word_index), repeat(args)))
#             valid_output, valid_ventilation_duration = create_output_dataframe(valid_data, word_index, args)
#             test_output, test_ventilation_duration = create_output_dataframe(test_data, word_index, args)
        
    print("data ready")
    
    # create meta data
    if args.meta_data:
        meta_data = [" ".join(["1", str(idx), "1"]) for idx in word_index.values()]
        print("meta data ready")
    
    # write to output
    if not args.no_data:
        if not args.split_datasets:
            with open(os.path.join(args.output_directory, 'data.txt'), "w") as file:
                file.writelines("\n".join(output))
            print("data written to", os.path.join(args.output_directory, 'data.txt'))
        else:
            with open(os.path.join(args.output_directory, 'train_data.txt'), "w") as file:
                file.writelines("\n".join(train_output))
            with open(os.path.join(args.output_directory, 'validation_data.txt'), "w") as file:
                file.writelines("\n".join(valid_output))
            with open(os.path.join(args.output_directory, 'test_data.txt'), "w") as file:
                file.writelines("\n".join(test_output))
            print("data written to", os.path.join(args.output_directory, 'train_data/validation_data/test_data.txt'))
    if args.meta_data:
        with open(os.path.join(args.output_directory, 'meta.txt'), "w") as file:
            file.writelines("\n".join(meta_data))
        print("meta data written to", os.path.join(args.output_directory, 'meta.txt'))
    if args.ventilation_duration:
        if not args.split_datasets:
            with open(os.path.join(args.output_directory, 'vent.txt'), "w") as file:
                file.writelines("\n".join(ventilation_duration))
            print("ventilation duration written to", os.path.join(args.output_directory, 'vent.txt'))
        else:
            with open(os.path.join(args.output_directory, 'train_vent.txt'), "w") as file:
                file.writelines("\n".join(train_ventilation_duration))
            with open(os.path.join(args.output_directory, 'validation_vent.txt'), "w") as file:
                file.writelines("\n".join(valid_ventilation_duration))
            with open(os.path.join(args.output_directory, 'test_vent.txt'), "w") as file:
                file.writelines("\n".join(test_ventilation_duration))
            print("ventilation duration written to", os.path.join(args.output_directory, 'train_vent/validation_vent/test_vent.txt'))
