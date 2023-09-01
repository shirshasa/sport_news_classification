## Roadmap


1. In notebook 1-initial_EDA_and_data_cleaning: 
    - initial study of a dataset and description of a classification task.
    - duplicates removal
   
2. In the notebook 2-EDA:
    - analysis of texts in the trainset

3. In the notebook 3-tf-idf-baseline-model-evaluation-error-analysis:
    - dataset pre-processing
    - dataset splitting into train and validation sets (based on oid grouping, in the way, there is no intersection in oid in train and val sets)
    - preparation and selection of tf-idf features
    - Naive Bayes classifier evaluation
    - Wrong predictions analysis

Evaluation results on a validation set:
- Accuracy: 0.77
- F1-macro score: 0.77

TODO:
1. Run advanced text cleaning.
2. Add per class confidence thresholds for baseline model.
3. Prepare filtering for adds and comments (it may be another classification model).
4. Experiment with DL models: Bert, transformers.
