# Mining heterogeneous clinical notes by multi-modal latent topic model

## Abstract

Latent knowledge can be extracted from electronic notes recorded during patient encounters with the health system. Using these clinical notes to decipher a patient’s underlying comorbidites, symptom burden, and treatment course is an ongoing challenge. Latent topic model as an efficient Bayesian method can be used to model each patient’s clinical notes as “documents” and the words in the notes as “tokens”. However, standard latent topic models assume that all of the notes follow the same topic distribution, regardless of the type of note or the domain expertise of the author. We propose a novel application of latent topic modeling, using multi-modal topic model to jointly infer distinct topic distributions of notes of different types. We applied our model to clinical notes from the MIMIC-III dataset to infer distinct topic distributions over the physician and nursing note types. We observed a consistent improvement in topic interpretability using multi-modal modeling over the baseline model that ignores the note types. By correlating the patients’ topic mixture with hospital mortality and prolonged mechanical ventilation, we identified several diagnostic topics that are associated with poor outcomes. Because of its elegant and intuitive formation, we envision a broad application of our approach in mining multi-modality text-based healthcare information that go beyond clinical notes.

## Organization

This repository is organized by task. The code for mechanical ventilation prediction is in the highest-level folder, and the code for mortality prediction and the simulation study are in their respective folders.

## MIMIC data preparing

This project makes use of the MIMIC-III dataset to predict the duration of mechanical ventilation for patients.

This code assumes you have the cohort file and the notes file handy. MIMIC-III is a restriced-access dataset, please consult their [website](https://physionet.org/content/mimiciii/1.4/) to gain credential.

In order to do so, the MIMIC-III dataset needs to be cleaned and filtered to include information from a chosen cohort of patients. 

The cohort was chosen based on the following pipeline:

![](/images/cohort_selection_template.png)


To filter the dataset, run the following command:

`python cohort_filtering.py --cohort_file "path/to/cohort/file" --notes_file "path/to/notes/file" --categories [list, of, note, categories] --verbosity 1/0`

## Data pre-processing

After the correct cohort is generated, the raw `.csv` file needs to be processed into a specific format for our topic models.

Specifically, this is done by the pythons scripts with prefix `note_to_training`.

For example, `note_to_training.py` prepare multi-type data and generate corresponding meta data. It also has the option of using pre-defined vocabulary, generating (binarized) MV outcome, select certain note types, splitting into train/valid/test sets and ignoring held-out admission IDs. It could be run like this:

``` bash
python note_to_training.py \
    {data file} \
    {output dir} \
    --max_df {maximal document frequency} \
    --use_vocab {vocabulary to use} \
```

For detailed description of each argument's functionality, run `python note_to_training.py --help`.

There is subtle differences between different versions of the script, as listed below:

- `note_to_training_singledatatype.py`: treat all tokens as one note type
- `note_to_training_singledatatype_diff.py`: treat all tokens as one note type, but differentiate tokens from different tokens by treating them as different tokens
- `mortality/note_to_training_mor*.py`: the same as `note_to_training*.py`, but generate mortality labels instead of MV outcomes

In the output folder as specified, the data file is named `data.txt` and meta data `meta.txt`, which are essential for running our topic model.

## Topic modeling

For topic modeling, we use the implementation of [MixEHR](https://github.com/li-lab-mcgill/mixehr). For instruction of running MixEHR, please refer to their repository.

## Simulation study

In order to study the performance of our inference algorithm, we run experiments on simulated datasets with various controlled parameters (such as the number of patients, number of tokens, vocabulary size and the number of note types). We then correlate the topic mixture learned from this simulated data with the ground truth topics and calculate the correlation between the two. If our inference algorithm is capable of recovering the hidden topics from documents, the recovered topics should strongly correlate with the ground truth.

In order to generate simulated data, refer to `simulation/mixehr_simulation.ipynb`. This code can be used to simulate data for:

- `number of patients = [1000, 4000, 8000]`
- `number of tokens = [1000, 1500, 2000]`
- `vocabulary size = [2000, 2500, 3000]`
- `number of note-types = [1, 2, 4]`

It generates the data files and meta files for running the topic model as mentioned in the above section. In order to compare the heterogenous_ehr model with the baseline LDA model, refer to `simulation/run_baselines.ipynb` which converts the multi-note type data to single-note type. This modified data can then be used for running LDA topic model.

On obtaining the inferred patient-topic mixture (theta) and topic-vocabulary distribution (phi) after training the topic model, the correlation with the ground truth can be calculated by running `python simulation/find_pearson_correlation_phi.py` and `python simulation/find_pearson_correlation_theta.py` for \phi and \theta respectively. The plots shown in the paper can be reproduced by referring to `simulation/plot_results.ipynb`. 
