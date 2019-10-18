import argparse
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.lm import NgramCounter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess notes, extract ventilation duration, and prepare for running MixEHR')
    parser.add_argument('input_file_path', help='path to the original file')
    parser.add_argument('output_directory', help='directory to store outputs')
    parser.add_argument('--no_data', action='store_true', help='NOT output data')
    parser.add_argument('--no_vocab', action='store_true', help='NOT store vocabulary')
    parser.add_argument('-m', '--meta_data', action='store_true', help='output meta data (default not)')
    parser.add_argument('-v', '--ventilation_duration', action='store_true', help='output ventilation duration (default not)')
    parser.add_argument('-s', '--split_datasets', action='store_true', help='split the whole cohort into train/validation/test (default not)')
    return parser.parse_args()

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
    sw = set(stopwords.words('english'))
    no_sw = " ".join([word for word in no_num.split() if word not in sw])
    return no_sw

def get_unigram_counts(text):
    text_word = [sentence.split() for sentence in text]
    text_unigram = [ngrams(sentence, 1) for sentence in text_word]
    return NgramCounter(text_unigram)

if __name__ == '__main__':
    args = parse_args()
    if args.no_vocab and args.no_data and not args.meta_data and not args.ventilation_duration:
        raise Exception('no output specified')

    data = pd.read_csv(args.input_file_path)
    print("data read")

    # merge notes of same HADM_ID
    merged_data = data[['SUBJECT_ID', "HADM_ID"]].drop_duplicates()
    merged_data["ALLTEXT"] = merged_data.HADM_ID.apply(lambda hadm_id: "\n".join(data[data.HADM_ID == hadm_id].TEXT))
    merged_data.dropna()
    print('after merging notes of same HADM_ID, we have', merged_data.shape[0], 'unique HADM_IDs')

    # preprocess notes
    merged_data["PROCTEXT"] = merged_data.ALLTEXT.apply(lambda text: preprocess_text(text))
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
        train_data.to_csv(os.path.join(args.output_directory, 'train.txt'), header=False, index=False)   
        valid_data.to_csv(os.path.join(args.output_directory, 'validation.txt'), header=False, index=False) 
        test_data.to_csv(os.path.join(args.output_directory, 'test.txt'), header=False, index=False) 
        print('train/valid/test sets written to', args.output_directory)
    else:
        train_data = merged_data
        
    # find words to drop
    vectorizer = CountVectorizer(min_df=5)
    _ = vectorizer.fit_transform(train_data['PROCTEXT'])
    word_index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
    
#     train_data['UNIQ_WORDS'] = train_data.PROCTEXT.apply(lambda text: ' '.join(list(set(text.split()))))
#     word_counts = get_unigram_counts(train_data['UNIQ_WORDS'])
#     word_index = {} # dict for word index
#     current_index = 0   # store current maximal index
#     for sentence in train_data["UNIQ_WORDS"].tolist():
#         for word in set(sentence.split()):
#             if word_counts[word] < 5:
#                 continue
#             if word in word_index.keys():
#                 index = word_index[word]
#             else:
#                 index = current_index + 1
#                 current_index += 1
#                 word_index[word] = index
    print('word indexes generated')

    # store vocabulary
    if not args.no_vocab:
        word_index_output = [','.join([word, str(idx)]) for word, idx in word_index.items()]
        with open(os.path.join(args.output_directory, 'vocab.txt'), "w") as file:
            file.writelines('\n'.join(word_index_output))
        print('vocabulary written to', os.path.join(args.output_directory, 'vocab.txt'))
            
    # create output dataframe
    ids = train_data["HADM_ID"].unique().tolist()  # HADM_IDs
    output = []
    ventilation_duration = []
    for idx, id in tqdm(enumerate(ids)):
        set_ventilation = False # indicate whether ventilation duration is calculated
        if not args.no_data or args.ventilation_duration:
            unigram_counts = get_unigram_counts(train_data[train_data.HADM_ID == id]["PROCTEXT"].tolist())
            for word in set(train_data[train_data.HADM_ID == id]["PROCTEXT"].values[0].split()):
                if word not in word_index.keys():
                    continue
                output.append(" ".join([str(idx), "1", str(word_index[word]), "0", str(unigram_counts[word])]))
                if args.ventilation_duration and not set_ventilation:
                    set_ventilation = True
                    ventilation_duration.append(" ".join([str(idx), str(train_data[train_data.HADM_ID == id]["DURATION"].values[0])]))
    print("data ready")
    
    # create meta data
    if args.meta_data:
        meta_data = [" ".join(["1", str(idx), "1"]) for idx in word_index.values()]
        print("meta data ready")
    
    # write to output
    if not args.no_data:
        with open(os.path.join(args.output_directory, 'data.txt'), "w") as file:
            file.writelines("\n".join(output))
        print("data written to", os.path.join(args.output_directory, 'data.txt'))
    if args.meta_data:
        with open(os.path.join(args.output_directory, 'meta.txt'), "w") as file:
            file.writelines("\n".join(meta_data))
        print("meta data written to", os.path.join(args.output_directory, 'meta.txt'))
    if args.ventilation_duration:
        with open(os.path.join(args.output_directory, 'vent.txt'), "w") as file:
            file.writelines("\n".join(ventilation_duration))
        print("ventilation duration written to", os.path.join(args.output_directory, 'vent.txt'))
