# mimic_mechanical_ventillation

This project makes use of the MIMIC-III dataset to predict the duration of mechanical ventilation for patients.

It makes use of mixEHR to learn meaningful topics from the notes data and uses these topics to train a classifier that predicts the mechanical ventilation duration.

In order to do so, the MIMIC-III dataset needs to be cleaned and filtered to include information from a chosen cohort of patients. 

The cohort was chosen based on the following pipeline:

![](/images/cohort_selection_template.png)


To filter the dataset, run the following command:

> python cohort_filtering.py --cohort_file "path/to/cohort/file" --notes_file "path/to/notes/file" --categories [list, of, note, categories] --verbosity 1/0

This code assumes you have the cohort file and the notes file handy.

This script does not give the labels of mechanical ventilation. That is handled by the `note_to_training.py` code. 