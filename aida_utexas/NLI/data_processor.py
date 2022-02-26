import csv
import pickle
import numpy as np

# read in input data 
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = [line for line in reader]
        return lines

    @classmethod
    def _read_csv(cls, input_file):
        with open(input_file) as f:
            reader = csv.reader(f)
            lines = [line for line in reader]
        return lines

    @classmethod
    def _read_pkl(cls, input_file):
        with open(input_file, 'rb') as f:
            lines = pickle.load(f)
        return lines


class EvalProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        # not used
        return self._read_csv(data_dir)
        
    def get_dev_examples(self, data_dir):
        # not used
        return self._read_csv(data_dir)

    def get_test_examples(self, data_dir):
        lines = self._read_csv(data_dir)
        pairs = [line for line in lines]
        return pairs[1:]

    def get_labels(self):
        return ["contradiction", "neutral", "entailment"]
