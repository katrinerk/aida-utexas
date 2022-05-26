'''
Built on pre-trained RoBERTa-Large NLI model, trained on MNLI dataset

Not used in evaluation
'''

import os
import sys
import csv
import logging
import torch
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

logger = logging.getLogger(__name__)

#######
# data processor to read nli_input.csv data
# return: header, data
class EvalProcessor(object):
    def get_examples(self, data_dir):
        lines = self._read_csv(data_dir)
        pairs = [line for line in lines]
        return pairs[0], pairs[1:]

    def get_labels(self):
        # pre-defined based on MNLI dataset label order
        return ["contradiction", "neutral", "entailment"]

    @classmethod
    def _read_csv(cls, input_file):
        with open(input_file) as f:
            reader = csv.reader(f)
            lines = [line for line in reader]
        return lines


#############
# data loader
class EvalDataset(Dataset):
    # load the dataset
    def __init__(self, lines, encoder):
        pairs = [[self._truncate(line[2]), line[5]] for line in lines]
        # tokenize claims
        self.X = collate_tokens([encoder.encode(pair[0], pair[1]) for pair in pairs], pad_idx=1)
        # arbitrarily 0 for evaluation
        self.Y = [0 for line in lines]
        # original data with id, filename, original claim sentence
        self.data = lines
    
    ########
    # truncate claim/query with format "somebody claims ....."
    # only keep sentence after "claims"
    def _truncate(self, query):
        if "claims" in query:
            idx = query.index("claims")
            temp = " ".join(query[idx+1:])
            return temp.capitalize()
        return query

    # number of pairs
    def __len__(self):
        return len(self.X)
 
    # get the next query/claim or claim/claim pair
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return [x, y]


class NLI_predicter():
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        # original data with ID, filename information
        self.data = self.dataloader.dataset.data

    def predict(self, batch_size, labels):
        num_written_lines = 0
        outputs = []
        for i, (x, _) in enumerate(self.dataloader):
            
            output = self.model(x, features_only=True, classification_head_name='mnli')
            output = nn.functional.softmax(output[0], dim=-1).detach().numpy()
            label = [labels[np.argmax(i)] for i in output]

            for i in range(batch_size):
                line = []
                line.append(label[i])
                for prob in output[i]:
                    line.append(prob)
                outputs.append(line)

            num_written_lines += batch_size
            logger.info('Predict {}'.format(num_written_lines))

        total_data_lines = len(self.data)
        assert num_written_lines == total_data_lines

        logger.info('Finish predict')
        return outputs


    def write_output(self, output_path, outputs, header):
        cols = header + ["nli_label", "contradict_prob", "neutral_prob", "entail_prob"]
        df = pd.DataFrame(columns = cols)

        for i, output in enumerate(outputs):

            tmp = []
            line = list(self.data[i]) 

            # add label and probabilities
            for ele in output:
                line.append(ele)
            
            tmp.append(pd.DataFrame([np.array(line)], columns = cols))

            buf = pd.concat(tmp, ignore_index=True)
            df = pd.concat([df, buf], ignore_index=True)

        df.to_csv(output_path, index=False)
        logger.info('Finish writing outputs to local')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=False, default='model.pt',
                        help="model checkpoint file")

    parser.add_argument('--batch', type=int, required=False, default=1, 
                        help="please ensure dataset size is multiple of batch size")

    parser.add_argument('--run', type=str, required=True, help="for example: ta2_colorado")

    parser.add_argument('--type', type=str, required=True, help="query_claim or claim_claim")

    parser.add_argument('--condition', type=str, required=True, 
                        help="condition5, contition6, condition7")
    
    parser.add_argument('--workspace', type=str, required=True, help="path the directory for processing work")
    
    args = parser.parse_args()
    
    # sanity check on condition
    if args.condition not in ["condition5", "condition6", "condition7"]:
        print("Error: need a condition that is condition5, condition6, condition7")
        sys.exit(1)
        
    # sanity check on work type
    if args.type not in ["query_claim", "claim_claim"]:
        print("Error: need a type that is query_claim or claim_claim")
        sys.exit(1)


    if args.type == "query_claim":
        input_file = Path(os.path.join(args.workspace, args.run, args.condition, "step2_query_claim_nli/nli_input.csv"))
        output_dir = os.path.join(args.workspace, args.run, args.condition, "step2_query_claim_nli")
        if not Path(output_dir).exists():
            #os.mkdir(output_dir)
            os.makedirs(output_dir)
        output_path = Path(os.path.join(output_dir, "q2d_nli.csv"))
    elif args.type == "claim_claim":
        input_file = Path(os.path.join(args.workspace, args.run, args.condition, "step2_query_claim_nli/claim_claim.csv"))
        output_dir = os.path.join(args.workspace, args.run, args.condition, "step2_query_claim_nli")
        if not Path(output_dir).exists():
            os.makedirs(output_dir)
        output_path = Path(os.path.join(output_dir, "d2d_nli.csv"))

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: {}'.format(device))
    if device.type == 'cuda':
        logger.info('Device name: {}'.format(torch.cuda.get_device_name(0)))

    # load roberta model
    roberta = RobertaModel.from_pretrained('roberta.large.mnli', checkpoint_file='model.pt')
    model = roberta.model
    model.eval()    # set model to evaluation state (delete dropout etc)

    # optional, load other checkpoint of roberta 
    if args.ckpt != 'model.pt':
        logger.info('Load check point')
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info('Create predict data generator')
    # get header, data, labels
    processor = EvalProcessor()
    header, data = processor.get_examples(input_file)
    labels = processor.get_labels()

    # create data loader for prediction
    predictset = EvalDataset(data, roberta)
    predictloader = torch.utils.data.DataLoader(predictset, batch_size=args.batch, drop_last=True)

    # start prediction
    logger.info('Total data: {}'.format(str(len(predictset))))
    logger.info('Start Prediction')
    predicter = NLI_predicter(model, predictloader)
    outputs = predicter.predict(args.batch, labels)

    # write output
    predicter.write_output(output_path, outputs, header)

    
if __name__ == '__main__':
    main()
