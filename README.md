This is the repository for my science fair project

Preprocess.py is for data pre-processing
It converts molecules from the text-based SMILES representation into a number-based fingerprint representation
Then it used scaffold split to split the data into train, test, and validation
Randomly choosed 50 positive and 50 negative cases from each split
Applies MinMax Scaling
Applies Principal Component Analysis(PCA)

quantum_model_datareuploading is the VQC training code
This is VQC model implemented with datareuploading so I can encode more features than my qubits 
