import os
import logging
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processor import *

logger = logging.getLogger(__name__)


class EvalDataset(Dataset):
    # load the dataset
    def __init__(self, lines):
        self.X = [[self._truncate(line[2]), line[5]] for line in lines]
        self.Y = [0 for line in lines]
        self.data = lines
 
    def _truncate(self, query):
        if "claims" in query:
            idx = query.index("claims")
            temp = " ".join(query[idx+1:])
            return temp.capitalize()
        return query

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return [x, y]


class NLI_predicter():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, batch_size, dataloader, labels, max_length=256):

        num_written_lines = 0
        outputs = []
        for i, (x, _) in enumerate(dataloader):

            tokenized_input_seq_pair = self.tokenizer.encode_plus(x[0][0], x[1][0],
                                                    max_length=max_length,
                                                    return_token_type_ids=True, truncation=True)

            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)

            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

            output = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)

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

        logger.info('Finish predict')
        return outputs


def write_output(data_dir, outputs, dataloader, labels, threshold):
    cols = ['Query_Filename', 'Query_ID', 'Query_Sentence', 'Claim_Filename', 'Claim_ID', 'Claim_Sentence', 'Related or Unrelated', 'Score', "nli_label", "contradict_prob", "neutral_prob", "entail_prob", 'adjust_nli_label'] 
    df = pd.DataFrame(columns = cols)

    for i, output in enumerate(outputs):
        info = dataloader.dataset.data[i] # [Query_Filename, Query_ID, Query_Sentence, Claim_Filename, Claim_ID, Claim_Sentence, Redundant_or_Independent, Score]
    
        tmp = []
        line = list(info)

        # add label and probabilities
        for ele in output:
            line.append(ele)
        
        # add adjust label
        probs = output[1:]
        if np.max(probs) < threshold:
            line.append('neutral')
        else:
            line.append(labels[np.argmax(probs)])
        tmp.append(pd.DataFrame([np.array(line)], columns = cols))

        buf = pd.concat(tmp, ignore_index=True)
        df = pd.concat([df, buf], ignore_index=True)

    df.to_csv(data_dir, index=False)
    logger.info('Finish writing outputs to local')


def main():
    parser = argparse.ArgumentParser()

    #TODO
    parser.add_argument('--batch', type=int, required=False, default=1, 
                        help="currectly only support batch size 1")

    parser.add_argument('--threshold', type=float, required=False, default=0.96,
                        help="the threshold used to separate neutral from entail and contradict")

    parser.add_argument('--data', type=str, required=True, help="for example: ta2_colorado")

    parser.add_argument('--input_file', type=str, required=False, default="../../evaluation_2022/dryrun_data/working",
                        help="path to related matched query-claim pair file")
    
    parser.add_argument('--output_path', type=str, required=False, default = "../../evaluation_2022/dryrun_data/working", 
						help="path to working space for output result")
    
    args = parser.parse_args()

    input_file = os.path.join(args.input_file, args.data, "condition5/step2_query_claim_nli/nli_input.csv")
    output_path = os.path.join(args.output_path, args.data, "condition5/step2_query_claim_nli/q2d_nli_combined.csv")

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: {}'.format(device))
    if device.type == 'cuda':
        logger.info('Device name: {}'.format(torch.cuda.get_device_name(0)))

    # load roberta model
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    logger.info('Create predict data generator')
    # read in input dataset
    processor = EvalProcessor()
    data = processor.get_test_examples(input_file)
    labels = ["entailment", "neutral", "contradiction"]

    # create data generator for prediction
    predictset = EvalDataset(data)
    predictloader = torch.utils.data.DataLoader(predictset, batch_size=args.batch, drop_last=True)

    # start prediction
    logger.info('Total data: {}'.format(str(len(predictset))))
    logger.info('Start Prediction')
    predicter = NLI_predicter(model, tokenizer)
    outputs = predicter.predict(args.batch, predictloader, labels)

    # write output
    write_output(output_path, outputs, predictloader, labels, args.threshold)

    
if __name__ == '__main__':
    main()