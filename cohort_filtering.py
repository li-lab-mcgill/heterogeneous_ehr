## Author: Pratheeksha Nair
## Date: 1/11/2019

########
# This file contains the code to pre-process the MIMIC-III dataset for the task of mechanical ventialtion prediction 
# The preprocessing steps involved are described in the function pre_process()
# This code assumes that you have identified a cohort of patients whose notes you wish to process and have access to this cohort in the form of a table
# This code also assumes that you have access to the MIMIC-III NOTEEVENTS dataset
# For more information, please refer to the main README.md file
########

import pandas as pd
import numpy as np
import argparse


# Reading in the data files


#def def_args():
# defining the arguments to run this code

parser = argparse.ArgumentParser()
parser.add_argument("--cohort_file", help="path to the cohort information")
parser.add_argument("--notes_file", help="path to the notes file (NOTEVENTS)")
parser.add_argument("--save_path", help="path to save file including filename")
parser.add_argument("--categories", help="list of note categories to be included. Eg, ['physician', 'nursing']")
parser.add_argument("--verbosity", help="set to 1 (default) for verbose descriptions of code, else set to 0", default=1)

args = parser.parse_args()
#	return args




def get_time_diff(merged):
# function that calculates the difference between the start of mechanical ventilation and the chart time of notes
# Input : Data table which is a merge between the selected cohort and the notes
# Output: List of time difference between the charttime and start of mechanical ventilation

    from datetime import datetime, timedelta
    from tqdm import tqdm_notebook as tqdm
    
    FMT = '%Y-%m-%d %H:%M:%S'
    time_from_vent_list = []

    for ind,rows in tqdm(merged.iterrows(), total= merged.shape[0]):

        # take difference
        time_from_vent = datetime.strptime(rows.CHARTTIME, FMT)- datetime.strptime(rows.FIRST_VENT_STARTTIME, FMT)

        # convert to hourr and add to list
        time_from_vent_list.append(t_from_vent.total_seconds()/timedelta (hours=1).total_seconds())
        
    return time_from_vent_list




def pre_process(cohort, notes, categories, verbosity=1):
# This function carries out the preprocessing of the data. It uses a pipeline as follows:
#        -- Consider only the first ICU stay of patients
#        -- Remove all rows from the notes table with invalid CHARTTIME and HADM_ID
#        -- Merge the cohort table with the notes table on the HADM_ID
#        -- Find the difference between chart time of noted and start of mechanical ventilation
#        -- Extract notes within the first 48 hours of mechanical ventilation
#        -- From this, take only those notes within the required categories

# Input: cohort table, notes table, list of categories, verbosity value
# Output: the pre-processed dataframe following the above pipeline
    
    if verbosity:
        print("Initial size of cohort: " + str(cohort.shape) + "\n")
        print("Initial size of notes table: " + str(notes.shape) + "\n")
        print("\n...............\n")
        print("Obtaining records corresponding to first ICU stay only\n")
        print("Removing rows with invalid admission IDs and null chart time of notes")
        print("Merging cohort with notes\n")
        
    # getting patient records of first icu stay only 
    first_icu_stay = cohort[cohort.FIRST_ICU_STAY == 't']
    
    # remove rows without CHARTTIME and HADM_ID values
    notes_df_filtered = notes[notes.CHARTTIME.notnull() & notes.HADM_ID.notnull()]
    
    
    # merging cohort with notes
    merged = pd.merge(notes_df_filtered, first_icu_stay[['HADM_ID','ICUSTAY_ID','FIRST_VENT_STARTTIME']], on = ['HADM_ID'], how='left')
    merged = merged[merged.FIRST_VENT_STARTTIME.notnull()]
    merged.drop_duplicates(subset='ROW_ID',keep='first',inplace=True)
    if verbosity:
        print("Size of merged table: " + str(merged.shape))
        print("Number of unique admissions " + str(len(merged.HADM_ID.unique())) + "\n")
        print("\n.................\n")
    
    # calculate the difference between chart time of notes and start of mechanical ventilation
    print("Calculating the difference between chart time of notes and start of mechanical ventilation \n")
    t_from_vent_list = get_time_diff(merged)
    merged['TIME_FROM_VENT'] = t_from_vent_list
    if verbosity:
        print(merged.columns)
    
    # getting notes from only first 48 hours
    merged_48hrs = merged[(merged.TIME_FROM_VENT <= 48) & (merged.TIME_FROM_VENT >= 0)]
    if verbosity:
        print("Take notes only from first 48 hours of mechanical ventilation \n")
        print("Number of unique admissions " + str(len(merged_48hrs.HADM_ID.unique())) + "\n")
    
    # retaining only notes of required categories
    res_notes = merged_48hrs[merged_48hrs.CATEGORY.isin(categories)]
    if verbosity:
        print("Retain notes of only required categories " + str(categories) + "\n")
        print("Final cohort size: " + str(res_notes.shape) +"\n")
        print("Final number of unique admissions "+ str(len(res_notes.HADM_ID.unique())) +"\n")
    
    return res_notes




def main():
	
	#args = def_args()

	cohort = pd.read_csv(args.cohort_file)
	notes_df = pd.read_csv(args.notes_file)
	output_path = args.save_path
	categories = args.categories
	verbosity = args.verbosity

	preprocessed_notes = pre_process(cohort, notes_df, categories, verbosity)

	preprocessed_notes.to_csv(output_path, index=False)
