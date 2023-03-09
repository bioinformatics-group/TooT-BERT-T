# TooT-BERT-T

TooT-BERT-T is a tool that predicts transmembrane transporter proteins using ProtBERT-BFD and logistic regression. This tool takes protein sequences in fasta format as input and outputs the predicted labels for sequence ID - transporter, nontransporter. The method used in this tool is described in the paper "TooT-BERT-T: A BERT Approach on Discriminating Transport Proteins from Non-transport Proteins" by H. Ghazikhani and G. Butler, published in the proceedings of the 16th International Conference on Practical Applications of Computational Biology and Bioinformatics (PACBB 2022).

## Installation

The list of required python packages is included in the file `**requirements.txt**`. To install these packages, run the following command:
```
pip install -r requirements.txt
```
## Usage

You can run the program as follows:
```
python run.py [input_fasta_file] [output_file]
```
For example:
```
python run.py test.fasta out.txt
```
where `**test.fasta**` is the input file containing protein sequences in fasta format and `**out.txt**` is the output file where the predicted labels will be written.

Note: This tool runs faster on a GPU.