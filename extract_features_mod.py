from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re
import csv
import sys
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from tqdm import tqdm, trange
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

from numpy import dot
from numpy.linalg import norm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, label):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids, label_id):
        # self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    # print("before", line)
                    line = list(unicode(cell, 'utf-8') for cell in line)
                    # print("after", line)
                lines.append(line)
            # print("hi",lines)
            lines  = lines[1:]
            labels = []
            for i in lines:
                if i[1] == "label" or i[1] in labels:
                    continue
                labels.append(i[1])
            # print("lines", lines)
            # print("labels", labels)

        return lines,labels

def initialise(lines, labels, num_sets_train, n, q, k, class_size):
    queries = []
    support = []
    classes = []
    temp_queries = []
    temp_support = []

    for i in range(num_sets_train):
        temp_classes = random.sample(labels, n)
        # print("Classes", temp_classes)
        classes = classes + temp_classes
        
        for i in temp_classes:
            j = labels.index(i)
            # print("hello", j, i, labels)
            query = np.random.choice(list(range(j*class_size, (j+1)*class_size)), q+k, False)
            temp_queries = temp_queries + query.tolist()[:q]
            temp_support = temp_support + query.tolist()[q:]
        # print(temp_support)
        # print(temp_queries)
    lines = np.asarray(lines)
    # print(support)
    lines_subset1 = lines[temp_support]
    support = lines_subset1.tolist()
    # random.shuffle(support)
    lines_subset2 = lines[temp_queries]
    queries = lines_subset2.tolist()
    # random.shuffle(queries)
    # print("classes", classes)
    # shuffled_support= []
    # shuffled_queries = []
    # for l in range(num_sets_train):
    #     temp_queries = []
    #     temp_queries = queries[l*(n*q):l*(n*q)+(n*q)]
    #     random.shuffle(temp_queries)
    #     shuffled_queries = shuffled_queries + temp_queries

    # for l in range(num_sets_train):
    #     temp_support = []
    #     temp_support = support[l*(n*k):l*(n*k)+(n*k)]
    #     random.shuffle(temp_support)
    #     shuffled_support= shuffled_support + temp_support
    # print("support", support, len(support))
    # print("queries", queries, len(queries))

    return classes, support, queries

class RelDataProcessor(DataProcessor):
    def get_dev_examples(self, input_file, num_sets_dev, n, q, k, class_size):
        """See base class."""
        lines, labels = self._read_tsv(input_file)
        classes, support, queries = initialise(lines, labels, num_sets_dev, n, q, k, class_size)
        return self._create_examples(support, "kjdn"), self._create_examples(queries, "kjdn"), classes, labels

    def get_labels(self):
        """See base class."""
        return self.classes

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_list = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            # text_b = line[4]
            label = line[1]
            examples.append(
                InputExample(unique_id=guid, text_a=text_a, text_b=None, label=label))
            #label_list.append(label)
        return examples

def convert_examples_to_features(examples, seq_length, tokenizer, label_list):
    """Loads a data file into a list of `InputFeature`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        label_id = label_map[example.label]
        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            # logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                # unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        # lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                # print("before", line)
                line = list(unicode(cell, 'utf-8') for cell in line)
                # print("after", line)
            # lines.append(line)
            # guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            # text_b = line[4]
            label = line[1]
            # print("hi",lines)
            # lines  = lines[1:]
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=None, label=label))
            unique_id += 1
    return examples

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    # print(preds, len(preds),labels, len(labels))
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    # parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # examples = read_examples(args.input_file)
    data_processor = RelDataProcessor()
    n = 5
    q = 20
    k = 5
    support, queries, classes, labels = data_processor.get_dev_examples(args.input_file, 200, n, q, k, 700)

    support_features = convert_examples_to_features(
        examples=support, seq_length=args.max_seq_length, tokenizer=tokenizer, label_list=labels)
    queries_features = convert_examples_to_features(examples=queries, seq_length=args.max_seq_length, tokenizer=tokenizer, label_list=labels)

    # unique_id_to_feature = {}
    # for feature in features:
    #     unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    label_map = {label : i for i, label in enumerate(labels)}

    class_ids = []
    for i in classes:
        class_id = label_map[i]
        class_ids.append(class_id)

    support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long)
    support_input_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long)
    support_label_ids = torch.tensor([f.label_id for f in support_features], dtype=torch.long)
    
    # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    queries_input_ids = torch.tensor([f.input_ids for f in queries_features], dtype=torch.long)
    queries_input_mask = torch.tensor([f.input_mask for f in queries_features], dtype=torch.long)
    queries_label_ids = torch.tensor([f.label_id for f in queries_features], dtype=torch.long)  
    
    # print(support_input_ids, support_input_mask, support_input_ids.shape, support_input_mask.shape)
    # print(queries_input_ids, queries_input_mask, queries_input_ids.shape, queries_input_mask.shape)
    support_data = TensorDataset(support_input_ids, support_input_mask, support_label_ids)
    queries_data = TensorDataset(queries_input_ids, queries_input_mask, queries_label_ids)
    # print(support_data, queries_data)

    # if args.local_rank == -1:
    #     eval_sampler = SequentialSampler(eval_data)
    # else:
    #     eval_sampler = DistributedSampler(eval_data)
    support_dataloader = DataLoader(support_data, batch_size=n*k)      
    queries_dataloader = DataLoader(queries_data, batch_size=n*q)

    model.eval()
    eval_loss = 0
    preds = []
    ground_truth = []
    nb_eval_steps = 0
    total_eval_loss=0
    
    for step, batch in enumerate(zip(queries_dataloader, support_dataloader)):

        classes_batch = class_ids[(step*n):((step*n)+n)]
        queries_input_ids, queries_input_mask, queries_label_ids= batch[0][0], batch[0][1], batch[0][2]
        support_input_ids, support_input_mask, support_label_ids= batch[1][0], batch[1][1], batch[1][2]

        class_map = {label : i for i, label in enumerate(classes_batch)}
        queries_label_ids1 = []

        for i in queries_label_ids.tolist():
            temp = class_map[i]
            queries_label_ids1.append(temp)

        queries_label_ids1 = torch.LongTensor(queries_label_ids1)
        queries_label_ids1 = queries_label_ids1.to(device)

        support_input_ids = support_input_ids.to(device)
        support_input_mask = support_input_mask.to(device)
        support_label_ids = support_label_ids.to(device)

        queries_input_ids = queries_input_ids.to(device)
        queries_input_mask = queries_input_mask.to(device)

        with torch.no_grad():

            encoded_layers_support, _ = model(support_input_ids, token_type_ids=None, attention_mask=support_input_mask)
            encoded_layers_queries, _ = model(queries_input_ids, token_type_ids=None, attention_mask=queries_input_mask)
            # print(encoded_layers_support.shape)
            # import ipdb; ipdb.set_trace()

            # print(encoded_layers_queries[-1].shape)
            # print(encoded_layers_support[-1].shape)

            encoded_layers_support = torch.sum(encoded_layers_support[-1], dim =1)
            # print(encoded_layers_support.shape)
            support_mask = torch.sum(support_input_mask, dim =1).repeat(encoded_layers_support.shape[1], 1).t().float()
            encoded_layers_support_avg = (encoded_layers_support/support_mask)

            encoded_layers_queries = torch.sum(encoded_layers_queries[-1], dim =1)
            queries_mask = torch.sum(queries_input_mask, dim =1).repeat(encoded_layers_queries.shape[1], 1).t().float()
            encoded_layers_queries_avg = (encoded_layers_queries/queries_mask)

            # import ipdb; ipdb.set_trace()
            div2 = torch.norm(encoded_layers_support_avg, dim=1)
            # div1 = norm(encoded_layers_queries_avg, axis=1).reshape(encoded_layers_queries_avg.shape[0], 1)
            div1 = torch.norm(encoded_layers_queries_avg, dim =1)
            # div2 = norm(encoded_layers_support_avg, axis=1).reshape(encoded_layers_support_avg.shape[0], 1)
            # import ipdb; ipdb.set_trace()

            # res1 = dot(encoded_layers_queries_avg, encoded_layers_support_avg.T)/dot(div1, div2.T)
            # res1= torch.from_numpy(res1)
            # res= res1.to(device)

            res = encoded_layers_queries_avg.mm(encoded_layers_support_avg.t())/torch.mul(div1.unsqueeze(1), div2.unsqueeze(1).t())
            s1 = []
            logits_support_max = torch.zeros([res.shape[0], n])

            for i in range(n): 
                d = (support_label_ids == classes_batch[i]).nonzero().view(-1).tolist()
                temp, _ = torch.max(res[:, d], dim = -1)
                s1.append(temp)

            logits_support_max = torch.stack(s1, dim =1)

        loss_fct = CrossEntropyLoss()
        eval_loss = loss_fct(logits_support_max.view(-1, n), queries_label_ids1)
        total_eval_loss += eval_loss
        nb_eval_steps += 1
        print("Validation loss", eval_loss)


        if len(preds) == 0:
            preds.append(logits_support_max.detach().cpu().numpy())
            ground_truth.append(queries_label_ids1.cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits_support_max.detach().cpu().numpy(), axis=0)
            ground_truth[0] = np.append(ground_truth[0], queries_label_ids1.cpu().numpy(), axis =0)

    preds = np.argmax(preds[0], axis=1)
    print("average validation loss of epoch", total_eval_loss/nb_eval_steps)
    print("validation accuracy", compute_metrics(preds, ground_truth[0]))

    # for input_ids, input_mask, example_indices in eval_dataloader:
    #     input_ids = input_ids.to(device)
    #     input_mask = input_mask.to(device)

    #     all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
    #     print(all_encoder_layers[-1].shape)            


if __name__ == "__main__":
    main()

