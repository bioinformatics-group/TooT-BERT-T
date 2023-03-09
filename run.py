import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import argparse
import joblib
from Bio import SeqIO
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Input FASTA file')
parser.add_argument('output_file', type=str, help='Output txt file with predicted labels')

args = parser.parse_args()

# We check if the input file is a fasta file. fasta file starts with > and then the id and then the sequence.
with open(args.input_file, 'r') as f:
    first_line = f.readline()
    if not first_line.startswith('>'):
        raise ValueError('Input file is not a fasta file.')

# Process the input file and write the output to output file.
with open(args.input_file, 'r') as f:
    # We read the input fasta file.
    records = list(SeqIO.parse(f, 'fasta'))
    # We create a list of sequences.
    sequences_ids = [(str(record.seq), str(record.id)) for record in records]

# We load the BERT model and the tokenizer.
print('Loading BERT model and tokenizer...')
tokenizer = BertTokenizer.from_pretrained('rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained('ghazikhanihamed/TransporterBERT')
model.to(device)

# We load the logistic regression model.
print('Loading logistic regression model...')
lr = joblib.load('lr_model.pkl')

# For each sequence, we tokenize it and then we pass it through the BERT model.
predictions = []
print("Sequence ID\t\tPredicted label")
print("------------\t\t---------------")
for sequence, id in sequences_ids:
    # Make space between each amino acid.
    sequence = ' '.join(sequence)
    # Replace unknown amino acids with X.
    sequence = sequence.replace('U', 'X')
    sequence = sequence.replace('O', 'X')
    sequence = sequence.replace('B', 'X')
    sequence = sequence.replace('Z', 'X')

    tokenized_sequence = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=20000, truncation=True)
    input_ids = torch.tensor([tokenized_sequence['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenized_sequence['attention_mask']]).to(device)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)[0]

    embedding = last_hidden_states[0].cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    seq_embedding = embedding[1:seq_len-1]
    mean_pool = np.mean(seq_embedding, axis=0)

    # We predict the label.
    prediction = lr.predict([mean_pool])
    predictions.append(f"Sequence:{id}\tPrediction:{prediction[0]}")

    # We print the id and the prediction.
    print(f"{id}\t{prediction[0]}")

# We write the output to the output file.
with open(args.output_file, 'w') as f:
    for prediction in predictions:
        f.write(prediction + '\n')

print('Finished.')