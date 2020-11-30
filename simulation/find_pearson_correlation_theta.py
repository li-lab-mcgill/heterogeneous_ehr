import numpy as np
import pandas as pd
import sys
from scipy.stats import pearsonr
import numpy as np
import pickle as pkl


def get_file_name(num_pats, num_tokens, vocab, iteration, t=None):
    # returns the names of simulated phi and inferred phi files
    folder_name = "simulated_data/"+str(num_pats)+"pats/"+str(num_tokens)+"tokens/"+str(vocab)+"vocab/"
    file_name_prefix = str(num_pats)+"pats_"+str(num_tokens)+"tokens_"+str(vocab)+"vocab"
    if t:
        sim_file_name = folder_name+"sim_theta_type"+str(t)+"_"+file_name_prefix+".csv"
        mxr_file_name = folder_name+"sim_mxr_data_type"+str(t)+"_"+file_name_prefix+"_sim_mxr_data_type"+str(t)+"_"+file_name_prefix+"_JCVB0_nmar_K50_iter"+str(iteration)+"_metaphe.csv"
    else:
        sim_file_name = folder_name+"sim_theta_drop_"+file_name_prefix+".csv"
        mxr_file_name = folder_name+"sim_mxr_data_drop_"+file_name_prefix+"_sim_mxr_data_drop_"+file_name_prefix+"_JCVB0_nmar_K50_iter"+str(iteration)+"_metaphe.csv"

    return sim_file_name, mxr_file_name


def find_corr(matrix1, matrix2):
    try:
        assert matrix1.shape[0] == matrix2.shape[1]
    except:
        print("Matrix dimensions are incompatible:" + str(matrix1.shape) + " , " + str(matrix2.shape))
        exit()

    size = matrix1.shape[0]
    corr_matrix = np.zeros([size, size])
    print(matrix1.shape)
    print(matrix2.shape)
    for i in range(size):
        for j in range(size):
            pc = pearsonr(matrix1[i], matrix2[:,j])
            corr_matrix[i][j] = pc[0]

    return corr_matrix

def plot_for_drop():
    # this function returns the file for a box plot for different number of note types
    iters = 481 # change this variable to the number of iterations for which heterogenous_ehr was trained
    sim_file_name, mxr_file_name = get_file_name(num_pats=4000, num_tokens=1500, vocab=2500, iteration=iters)
    return pearson_corr(sim_file_name, mxr_file_name)

def plot_for_types():
    # this function returns the file for a box plot over different number of note-types. Here, the num_of_patients is 4000, num_of_tokens is 1500 and vocabulary size is 2500 
    #iters = {1:1000, 2:1000, 4:480}
    #iters = {1:406, 2:560, 4:806}
    #iters = {1:406, 2:802, 4:1000}
    #iters = {1:406, 2:541, 4:782}
    iters = {2:787, 4:1000} # update this dictionary for the number of iterations the model was trained for each note-type
    type_results = {}
    for t in iters.keys():
        sim_file_name, mxr_file_name = get_file_name(num_pats=4000, num_tokens=300, vocab=2500, iteration=iters[t], t=t)
        type_results[t] = pearson_corr(sim_file_name, mxr_file_name, range(1,t+1))

    return type_results

def plot_for_patients():
    # this function returns the file for a box plot over different number of patients. Here the num_of_tokens is fixed to 1500 and the vocabulary size is fixed to 2500
    iters = {1000:1000, 4000:560, 8000:517} # update this dictionary for the number of iterations the model was trained for each number of patients
    pat_results = {}
    for patients in [1000, 4000, 8000]:
        sim_file_name, mxr_file_name = get_file_name(num_pats=patients, num_tokens=1500, vocab=2500, iteration=iters[patients])
        pat_results[patients] = pearson_corr(sim_file_name, mxr_file_name)

    return pat_results

def plot_for_tokens():
    # this function returns the file for a box plot over different number of tokens. Here, num_of_patients is fixed to 4000 and the vocabulary size is fixed to 2500
    iters = {1000:677, 1500:560, 2000:547} # update this dictionary for the number of iterations the model was trained for each number of tokens
    token_results = {}
    for tokens in iters.keys():
        sim_file_name, mxr_file_name = get_file_name(num_pats=4000, num_tokens=tokens, vocab=2500, iteration=iters[tokens])
        token_results[tokens] = pearson_corr(sim_file_name, mxr_file_name)

    return token_results

def plot_for_vocab():
    # this function returns the file for a box plot over different vocabulary sizes. Here, num_of_patients is fixed to 4000 and the number of tokens is fixed to 1500
    iters = {2000:534, 2500:560, 3000:490} # update this dictionary for the number of iterations the model was trained for each vocab size
    vocab_results = {}
    for vocab in iters.keys():
        sim_file_name, mxr_file_name = get_file_name(num_pats=4000, num_tokens=1500, vocab=vocab, iteration=iters[vocab])
        vocab_results[vocab] = pearson_corr(sim_file_name, mxr_file_name)

    return vocab_results

def pearson_corr(sim_topics_file, mxr_topics_file, types=None):
    # function that calculates the pearson's correlation between inferred and ground-truth topics
    sim_topics = pkl.load(open(sim_topics_file, 'rb'))
    #sim_topics_df = pd.DataFrame(sim_topics)
    #sim_topics = pd.read_csv(sim_topics_file, header=None)

    mxr_topics = pd.read_csv(mxr_topics_file, header=None)
    #mxr_topics_df = mxr_topics_df[mxr_topics_df.columns[2:]].T
    pearson_correlations = {}

    if types == None:
        types = [1,2]

    p_corr = find_corr(np.transpose(sim_topics), mxr_topics.values)
    p_corr_max = np.max(p_corr, axis=0)

    pearson_correlations = p_corr_max

    return pearson_correlations
    
def main():

    pkl.dump(plot_for_vocab(), open("results/vocab_corr_theta.pkl","wb"))
    pkl.dump(plot_for_types(), open("results/types_corr_theta.pkl","wb"))
    pkl.dump(plot_for_drop(), open("results/drop_corr_theta.pkl","wb"))
    pkl.dump(plot_for_patients(), open("results/patients_corr_theta.pkl","wb"))
    pkl.dump(plot_for_tokens(), open("results/tokens_corr_theta.pkl","wb"))

main()
