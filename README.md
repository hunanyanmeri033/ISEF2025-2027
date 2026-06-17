# Evaluating Classical Preprocessing vs. Quantum Circuit Complexity in Hybrid Models for Molecular Activity Prediction

This is the repository for my science fair project

## Preprocess.py is for data pre-processing
[Preprocess.py](src/preprocess.py)

It converts molecules from the text-based SMILES representation into a number-based fingerprint representation

Then it used scaffold split to split the data into train, test, and validation

Randomly choosed 50 positive and 50 negative cases from each split

Applies MinMax Scaling

Applies Principal Component Analysis(PCA)

## quantum_model_datareuploading is the VQC training code
[quantum_model_datareuploading.py](src/quantum_model_datareuplodaing.py)

This is VQC model implemented with datareuploading so I can encode more features than my qubits 

## classical_model.py
[classical_model.py](src/classical_model.py)

This is the classical model for my project. It is an XGBoost model.

## quantum_data_reupload.py
[quantum_data_reupload.py](src/quantum_data_reupload.py)

This file is for measuring accuracy and ROC AUC values of a model after running quantum_model_datareuploading.py
