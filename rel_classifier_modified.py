from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from numpy import dot
from numpy.linalg import norm

logger = logging.getLogger(__name__)

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertForRelationClassification(BertPreTrainedModel):

    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertForRelationClassification, self).__init__(config)
        self.output_attentions = output_attentions
        # self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(0.2)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, head_mask=head_mask)
        encoded_layers, pooled_output = outputs
        encoded_layer = self.dropout(encoded_layers[-1])
        encoded_layer = self.LayerNorm(encoded_layer)
        # print(encoded_layers.shape)
        # logits = self.classifier(pooled_output)
        return encoded_layer, pooled_output

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            lines  = lines[1:]
            labels = []
            for i in lines:
                if i[1] == "label" or i[1] in labels:
                    continue
                labels.append(i[1])
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
            query = np.random.choice(list(range(j*class_size, (j+1)*class_size)), q+k, False)
            temp_queries = temp_queries + query.tolist()[:q]
            temp_support = temp_support + query.tolist()[q:]

    lines = np.asarray(lines)
    # print(support)
    lines_subset1 = lines[temp_support]
    support = lines_subset1.tolist()
    # random.shuffle(support)
    lines_subset2 = lines[temp_queries]
    queries = lines_subset2.tolist()
    # random.shuffle(queries)
    # print("classes", classes)
    shuffled_support= []
    shuffled_queries = []
    for l in range(num_sets_train):
        temp_queries = []
        temp_queries = queries[l*(n*q):l*(n*q)+(n*q)]
        random.shuffle(temp_queries)
        shuffled_queries = shuffled_queries + temp_queries

    for l in range(num_sets_train):
        temp_support = []
        temp_support = support[l*(n*k):l*(n*k)+(n*k)]
        random.shuffle(temp_support)
        shuffled_support= shuffled_support + temp_support
    # print("support", shuffled_support, len(shuffled_support))
    # print("queries", shuffled_queries, len(shuffled_queries))

    return classes, shuffled_support, shuffled_queries

# unique_classes, classes, support, queries = intialise("train_full.tsv", 100, 5, 20, 5, 700)
# print("unique classes", unique_classes, len(unique_classes))
# print("classes", len(classes))
# print("support", len(support))
# print("queries", len(queries))

class MultiClassClassification(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    # def __init__(self, input_file, n, q, k, class_size):
    #     self.labels, self.classes, self.support, self.queries = initialise(input_file, n, q, k, class_size)

    def get_train_examples(self, data_dir, num_sets_train, n, q, k, class_size):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_full.tsv")))
        lines, labels = self._read_tsv(os.path.join(data_dir, "train_full.tsv"))
        classes, support, queries = initialise(lines, labels, num_sets_train, n, q, k, class_size)
        return self._create_examples(support, "train"), self._create_examples(queries, "train"), classes, labels

    def get_dev_examples(self, data_dir, num_sets_dev, n, q, k, class_size):
        """See base class."""
        lines, labels = self._read_tsv(os.path.join(data_dir, "dev_full.tsv"))
        classes, support, queries = initialise(lines, labels, num_sets_dev, n, q, k, class_size)
        return self._create_examples(support, "dev"), self._create_examples(queries, "dev"), classes, labels

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
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            #label_list.append(label)
        return examples

def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer, label_list):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}


    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
            # logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        
        # if ex_index < 1:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    # print(preds, len(preds),labels, len(labels))
    # assert len(preds) == len(labels)
    # import ipdb; ipdb.set_trace()
    return {"acc": simple_accuracy(preds, labels)}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--N',
                        type=int,
                        default=5,
                        help="no of relations")
    parser.add_argument('--K',
                        type=int,
                        default=5,
                        help="no of support examples per relation")
    parser.add_argument('--Q',
                        type=int,
                        default=20,
                        help="no of queries per relation")
    parser.add_argument('--num_sets_train',
                        type=int,
                        default=200,
                        help="no of training sets")
    parser.add_argument('--num_sets_dev',
                        type=int,
                        default=100,
                        help="no of eval sets")
    parser.add_argument("--learning_rate",
                        default=1.5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.65,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # args.queries_batch_size = args.queries_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = MultiClassClassification()      
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForRelationClassification.from_pretrained(args.bert_model)
    
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # unique_classes, classes_list = processor.get_labels()
    # num_labels = len(label_list)

    if args.do_train:

    # Prepare dataloader    
        support_examples, queries_examples, classes, labels = processor.get_train_examples(args.data_dir, args.num_sets_train, args.N, args.Q, args.K, 700)
        support_features = convert_examples_to_features(support_examples, args.max_seq_length, tokenizer, labels)
        queries_features = convert_examples_to_features(queries_examples, args.max_seq_length, tokenizer, labels)
        # print(queries_features)
        assert len(queries_features) == (args.N * args.Q * args.num_sets_train)

        label_map = {label : i for i, label in enumerate(labels)}
        class_ids = []
        for i in classes:
            class_id = label_map[i]
            class_ids.append(class_id)

        support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long)
        support_input_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long)
        support_segment_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long)
        support_label_ids = torch.tensor([f.label_id for f in support_features], dtype=torch.long)
        # print(support_input_ids, support_label_ids)
        queries_input_ids = torch.tensor([f.input_ids for f in queries_features], dtype=torch.long)
        queries_input_mask = torch.tensor([f.input_mask for f in queries_features], dtype=torch.long)
        queries_segment_ids = torch.tensor([f.segment_ids for f in queries_features], dtype=torch.long)
        queries_label_ids = torch.tensor([f.label_id for f in queries_features], dtype=torch.long)  
        # print(queries_input_ids, queries_label_ids)
        support_data = TensorDataset(support_input_ids, support_input_mask, support_segment_ids, support_label_ids)
        queries_data = TensorDataset(queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids)

        # if args.local_rank == -1:
        #     support_sampler = RandomSampler(support_data)
        #     queries_sampler = RandomSampler(queries_data)
        # else:
        #     support_sampler = DistributedSampler(support_data).detach()
        #     queries_sampler = DistributedSampler(queries_data)

        support_batch_size = args.N * args.K 
        queries_batch_size = args.N * args.Q

        support_dataloader = DataLoader(support_data, batch_size=support_batch_size)      
        queries_dataloader = DataLoader(queries_data, batch_size=queries_batch_size)

        num_train_optimization_steps = len(queries_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

        # model_params = ["module.bert.encoder.layer.0.attention.self.query.weight","module.bert.encoder.layer.0.attention.self.query.bias","module.bert.encoder.layer.0.attention.self.key.weight","module.bert.encoder.layer.0.attention.self.key.bias","module.bert.encoder.layer.0.attention.self.value.weight","module.bert.encoder.layer.0.attention.self.value.bias","module.bert.encoder.layer.0.attention.output.dense.weight","module.bert.encoder.layer.0.attention.output.dense.bias","module.bert.encoder.layer.0.attention.output.LayerNorm.weight","module.bert.encoder.layer.0.attention.output.LayerNorm.bias","module.bert.encoder.layer.0.intermediate.dense.weight","module.bert.encoder.layer.0.intermediate.dense.bias","module.bert.encoder.layer.0.output.dense.weight","module.bert.encoder.layer.0.output.dense.bias","module.bert.encoder.layer.0.output.LayerNorm.weight","module.bert.encoder.layer.0.output.LayerNorm.bias","module.bert.encoder.layer.1.attention.self.query.weight","module.bert.encoder.layer.1.attention.self.query.bias","module.bert.encoder.layer.1.attention.self.key.weight","module.bert.encoder.layer.1.attention.self.key.bias","module.bert.encoder.layer.1.attention.self.value.weight","module.bert.encoder.layer.1.attention.self.value.bias","module.bert.encoder.layer.1.attention.output.dense.weight","module.bert.encoder.layer.1.attention.output.dense.bias","module.bert.encoder.layer.1.attention.output.LayerNorm.weight","module.bert.encoder.layer.1.attention.output.LayerNorm.bias","module.bert.encoder.layer.1.intermediate.dense.weight","module.bert.encoder.layer.1.intermediate.dense.bias","module.bert.encoder.layer.1.output.dense.weight","module.bert.encoder.layer.1.output.dense.bias","module.bert.encoder.layer.1.output.LayerNorm.weight","module.bert.encoder.layer.1.output.LayerNorm.bias","module.bert.encoder.layer.2.attention.self.query.weight","module.bert.encoder.layer.2.attention.self.query.bias","module.bert.encoder.layer.2.attention.self.key.weight","module.bert.encoder.layer.2.attention.self.key.bias","module.bert.encoder.layer.2.attention.self.value.weight","module.bert.encoder.layer.2.attention.self.value.bias","module.bert.encoder.layer.2.attention.output.dense.weight","module.bert.encoder.layer.2.attention.output.dense.bias","module.bert.encoder.layer.2.attention.output.LayerNorm.weight","module.bert.encoder.layer.2.attention.output.LayerNorm.bias","module.bert.encoder.layer.2.intermediate.dense.weight","module.bert.encoder.layer.2.intermediate.dense.bias","module.bert.encoder.layer.2.output.dense.weight","module.bert.encoder.layer.2.output.dense.bias","module.bert.encoder.layer.2.output.LayerNorm.weight","module.bert.encoder.layer.2.output.LayerNorm.bias","module.bert.encoder.layer.3.attention.self.query.weight","module.bert.encoder.layer.3.attention.self.query.bias","module.bert.encoder.layer.3.attention.self.key.weight","module.bert.encoder.layer.3.attention.self.key.bias","module.bert.encoder.layer.3.attention.self.value.weight","module.bert.encoder.layer.3.attention.self.value.bias","module.bert.encoder.layer.3.attention.output.dense.weight","module.bert.encoder.layer.3.attention.output.dense.bias","module.bert.encoder.layer.3.attention.output.LayerNorm.weight","module.bert.encoder.layer.3.attention.output.LayerNorm.bias","module.bert.encoder.layer.3.intermediate.dense.weight","module.bert.encoder.layer.3.intermediate.dense.bias","module.bert.encoder.layer.3.output.dense.weight","module.bert.encoder.layer.3.output.dense.bias","module.bert.encoder.layer.3.output.LayerNorm.weight","module.bert.encoder.layer.3.output.LayerNorm.bias"]
        
        # for name, param in model.named_parameters():
        #     if name in model_params:
        #         param.requires_grad = False 
                
        # print("MODEL PARAMETERS")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        
        ###########################################################VALIDATION SET######################################################################
        support_val, queries_val, classes_val, labels_val = processor.get_dev_examples(args.data_dir, 150, args.N, args.Q, args.K, 700)
        support_features_val = convert_examples_to_features(support_val, args.max_seq_length, tokenizer, labels_val)
        queries_features_val = convert_examples_to_features(queries_val, args.max_seq_length, tokenizer, labels_val)
        label_map_val = {label : i for i, label in enumerate(labels_val)}
        # print(label_map_val)
        class_ids_val = []
        for i in classes_val:
            class_id = label_map_val[i]
            class_ids_val.append(class_id)
        # print(class_ids_val)

        support_input_ids_val = torch.tensor([f.input_ids for f in support_features_val], dtype=torch.long)
        support_input_mask_val = torch.tensor([f.input_mask for f in support_features_val], dtype=torch.long)
        support_segment_ids_val = torch.tensor([f.segment_ids for f in support_features_val], dtype=torch.long)
        support_label_ids_val = torch.tensor([f.label_id for f in support_features_val], dtype=torch.long)
    # print(support_label_ids)
        queries_input_ids_val = torch.tensor([f.input_ids for f in queries_features_val], dtype=torch.long)
        queries_input_mask_val = torch.tensor([f.input_mask for f in queries_features_val], dtype=torch.long)
        queries_segment_ids_val = torch.tensor([f.segment_ids for f in queries_features_val], dtype=torch.long)
        queries_label_ids_val = torch.tensor([f.label_id for f in queries_features_val], dtype=torch.long)  
        # print(queries_label_ids_val1)
        support_data_val = TensorDataset(support_input_ids_val, support_input_mask_val, support_segment_ids_val, support_label_ids_val)
        queries_data_val = TensorDataset(queries_input_ids_val, queries_input_mask_val, queries_segment_ids_val, queries_label_ids_val)

        support_dataloader_val = DataLoader(support_data_val, batch_size=support_batch_size)      
        queries_dataloader_val = DataLoader(queries_data_val, batch_size=queries_batch_size)
    #     ###########################################################VALIDATION SET######################################################################
        total_val_loss = 0
        val_iter = 0
        preds_val= []
        ground_truth_val = []
    #####################################################################################validation before epoch 0#########################################################        
        model.eval()
        print("performance on validation set before training")
        for step, batch in enumerate(zip(queries_dataloader_val, support_dataloader_val)):
            preds_val_batch =[]
            ground_truth_batch = []
            class_batch_val = class_ids_val[(step*args.N):((step*args.N)+args.N)]
            queries_input_ids_val, queries_input_mask_val, queries_segment_ids_val, queries_label_ids_val= batch[0][0], batch[0][1], batch[0][2], batch[0][3]
            support_input_ids_val, support_input_mask_val, support_segment_ids_val, support_label_ids_val= batch[1][0], batch[1][1], batch[1][2], batch[1][3]
        
            class_map_val = {label : i for i, label in enumerate(class_batch_val)}
            queries_label_ids_val1 = []

            for i in queries_label_ids_val.tolist():
                temp = class_map_val[i]
                queries_label_ids_val1.append(temp)

            queries_label_ids_val1 = torch.LongTensor(queries_label_ids_val1)

            support_input_ids_val = support_input_ids_val.to(device)
            support_input_mask_val = support_input_mask_val.to(device)
            support_segment_ids_val = support_segment_ids_val.to(device)
            support_label_ids_val = support_label_ids_val.to(device)

            queries_input_ids_val = queries_input_ids_val.to(device)
            queries_input_mask_val = queries_input_mask_val.to(device)
            queries_segment_ids_val = queries_segment_ids_val.to(device)
            queries_label_ids_val1 = queries_label_ids_val1.to(device)

            with torch.no_grad():

                encoded_support_val, embs_support_val = model(support_input_ids_val, support_segment_ids_val, support_input_mask_val)
                encoded_queries_val, embs_queries_val = model(queries_input_ids_val, queries_segment_ids_val, queries_input_mask_val)
                # print(encoded_support_val.shape)
                # import ipdb; ipdb.set_trace()
                # encoded_support_val_cls = encoded_support_val[:,0,:]
                # encoded_queries_val_cls = encoded_queries_val[:,0,:]

                encoded_support_val = torch.sum(encoded_support_val, dim =1)
                # print(encoded_support_val.shape)
                support_mask_val = torch.sum(support_input_mask_val, dim =1).repeat(encoded_support_val.shape[1], 1).t().float()
                # print(encoded_support_val.shape)
                encoded_support_val_avg = (encoded_support_val/support_mask_val)
                encoded_queries_val = torch.sum(encoded_queries_val, dim =1)

                queries_mask_val = torch.sum(queries_input_mask_val, dim =1).repeat(encoded_queries_val.shape[1], 1).t().float()
                encoded_queries_val_avg = (encoded_queries_val/queries_mask_val)

                div1 = torch.norm(encoded_queries_val_avg, dim =1)
                # div1 = norm(encoded_queries_val_avg, axis=1).reshape(encoded_queries_val_avg.shape[0],1)
    
                div2 = torch.norm(encoded_support_val_avg, dim =1)
                # div2 = norm(encoded_support_val_avg, axis=1).reshape(encoded_support_val_avg.shape[0],1)
    
                # import ipdb; ipdb.set_trace()

                res_val = encoded_queries_val_avg.mm(encoded_support_val_avg.t())/torch.mul(div1.unsqueeze(1), div2.unsqueeze(1).t())
                # res1_val= torch.from_numpy(res1_val)
                # res_val= res1_val.to(device)

                s1_val = []
                logits_support_max_val = torch.zeros([res_val.shape[0], args.N])

                for i in range(args.N): 
                    d = (support_label_ids_val == class_batch_val[i]).nonzero().view(-1).tolist()
                    temp, _ = torch.max(res_val[:, d], dim = -1)
                    s1_val.append(temp)

                logits_support_max_val = torch.stack(s1_val, dim =1) 

            loss_fct = CrossEntropyLoss()
            val_loss = loss_fct(logits_support_max_val.view(-1, args.N), queries_label_ids_val1)
            total_val_loss +=val_loss
            val_iter+=1
        # val_result = compute_metrics(logits_support_max_val.detach().cpu().numpy(), queries_label_ids_val1.cpu().numpy())
        # classes_batch = class_ids_val[(step*args.N):((step*args.N)+args.N)]
            print("Validation loss", val_loss)
            if len(preds_val) == 0:
                preds_val.append(logits_support_max_val.detach().cpu().numpy())
                ground_truth_val.append(queries_label_ids_val1.cpu().numpy())
            else:
                preds_val[0] = np.append(preds_val[0], logits_support_max_val.detach().cpu().numpy(), axis=0)
                ground_truth_val[0] = np.append(ground_truth_val[0], queries_label_ids_val1.cpu().numpy(), axis =0)
            preds_val_batch.append(logits_support_max_val.detach().cpu().numpy())
            ground_truth_batch.append(queries_label_ids_val1.cpu().numpy()) 
            # print("validation accuracy for batch", compute_metrics(np.argmax(preds_val_batch[0]), ground_truth_batch[0]))
        # print(preds_val, ground_truth_val)
        preds_val = np.argmax(preds_val[0], axis=1)
        print("average validation loss of epoch", total_val_loss/val_iter)
        print("validation accuracy", compute_metrics(preds_val, ground_truth_val[0]))
    #####################################################################################validation before epoch 0#########################################################        
        running_loss = 0 
        count =0
        model.train()
        nb_tr_steps = 0
        tr_loss = 0

        ######################################start of training###############################################################

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            # tr_loss = 0
            # nb_tr_examples, nb_tr_steps = 0, 0
            avg_loss_epoch = 0
            num_iter_epoch = 0
            for step, batch in enumerate(zip(queries_dataloader, support_dataloader)):
                # print(batch, len(batch))
                classes_batch = class_ids[(step*args.N):((step*args.N)+args.N)]
                queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids= batch[0][0], batch[0][1], batch[0][2], batch[0][3]
                support_input_ids, support_input_mask, support_segment_ids, support_label_ids= batch[1][0], batch[1][1], batch[1][2], batch[1][3]
                # print("support",support_input_ids, support_label_ids)
                # print(queries_input_ids, queries_label_ids)

                # print(queries_label_ids, len(queries_label_ids), support_label_ids, len(support_label_ids))
                class_map = {label : i for i, label in enumerate(classes_batch)}
                # print(class_map)
                queries_label_ids1 = []
                for i in queries_label_ids.tolist():
                    temp = class_map[i]
                    queries_label_ids1.append(temp)

                # classes_batch1 = torch.arange(args.N)
                # queries_label_ids1 = classes_batch1.view(-1, 1).repeat(1, args.Q).view(-1)
                queries_label_ids1 = torch.LongTensor(queries_label_ids1)
                queries_label_ids1 = queries_label_ids1.to(device)

                support_input_ids = support_input_ids.to(device)
                support_input_mask = support_input_mask.to(device)
                support_segment_ids = support_segment_ids.to(device)
                support_label_ids = support_label_ids.to(device)

                queries_input_ids = queries_input_ids.to(device)
                queries_input_mask = queries_input_mask.to(device)
                queries_segment_ids = queries_segment_ids.to(device)
                # queries_label_ids = queries_label_ids.to(device)

                encoded_layers_support, embs_support = model(support_input_ids, support_segment_ids, support_input_mask)
                # print("embs support",embs_support)
                encoded_layers_queries, embs_queries = model(queries_input_ids, queries_segment_ids, queries_input_mask)
                # print("embs queries",embs_queries)
                # dims_support = list(logits_support.size())
                # logits_support_avg = torch.zeros([args.N, dims_support[1]])
                # logits_support_avg = logits_support_avg.to(device)

                # for i in range(len(classes_batch)):
                #     for j in range(dims_support[0]):
                #         if support_label_ids[j] == classes_batch[i]:
                #             logits_support_avg[i] = logits_support_avg[i] + logits_support[j]eno
                # # logits_support_avg[i] = logits_support_avg[i] / K
                # logits_support_avg = logits_support_avg/args.K
                # print("logits support average", logits_support_avg)
                
                # print(queries_label_ids1)
                # import ipdb; ipdb.set_trace()

                # encoded_layer_support_cls = encoded_layers_support[:,0,:]
                # encoded_layer_queries_cls = encoded_layers_queries[:,0,:]
                encoded_layers_support = torch.sum(encoded_layers_support, dim =1)
                # encoded_layers_support2 = Variable(encoded_layers_support1, requires_grad=True)
                support_mask = torch.sum(support_input_mask, dim =1).repeat(encoded_layers_support.shape[1], 1).t().float()
                # support_mask = Variable(support_mask1, requires_grad=True)
                encoded_layers_support_avg = (encoded_layers_support/support_mask)

                encoded_layers_queries = torch.sum(encoded_layers_queries, dim =1)
                # encoded_layers_queries2 = Variable(encoded_layers_queries1, requires_grad=True)
                queries_mask = torch.sum(queries_input_mask, dim =1).repeat(encoded_layers_queries.shape[1], 1).t().float()
                # queries_mask = Variable(queries_mask1, requires_grad=True)
                encoded_layers_queries_avg = (encoded_layers_queries/queries_mask)

                div2 = torch.norm(encoded_layers_support_avg, dim=1)
                # div1 = norm(encoded_layers_queries_avg, axis=1).reshape(encoded_layers_queries_avg.shape[0], 1)
                div1 = torch.norm(encoded_layers_queries_avg, dim =1)
                # div2 = norm(encoded_layers_support_avg, axis=1).reshape(encoded_layers_support_avg.shape[0], 1)

                res = encoded_layers_queries_avg.mm(encoded_layers_support_avg.t())/torch.mul(div1.unsqueeze(1), div2.unsqueeze(1).t())
                
                s1 = []
                logits_support_max = torch.zeros([res.shape[0], args.N])

                for i in range(args.N): 
                	d = (support_label_ids == classes_batch[i]).nonzero().view(-1).tolist()
                	temp, _ = torch.max(res[:, d], dim = -1)
                	s1.append(temp)

                logits_support_max = torch.stack(s1, dim =1) 
                # import ipdb; ipdb.set_trace()
                # print("res",res)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits_support_max.view(-1, args.N), queries_label_ids1)
                # loss = Variable(loss, requires_grad = True)

                # import ipdb; ipdb.set_trace()

                print("---------------train loss-------------",loss)

                if n_gpu > 1:
                    loss = loss.mean() 
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                avg_loss_epoch += loss.item()
                num_iter_epoch += 1

                # nb_tr_examples += queries_input_ids.size(0)
                  
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                preds_val = []
                ground_truth_val = []
                total_val_loss = 0
                val_iter =0
                ############################# Validation at 100 steps#####################################################################
                if nb_tr_steps % 20 ==0:
                    for step, batch in enumerate(zip(queries_dataloader_val, support_dataloader_val)):
                        class_batch_val = class_ids_val[(step*args.N):((step*args.N)+args.N)]
                        queries_input_ids_val, queries_input_mask_val, queries_segment_ids_val, queries_label_ids_val= batch[0][0], batch[0][1], batch[0][2], batch[0][3]
                        support_input_ids_val, support_input_mask_val, support_segment_ids_val, support_label_ids_val= batch[1][0], batch[1][1], batch[1][2], batch[1][3]
                    
                        class_map_val = {label : i for i, label in enumerate(class_batch_val)}
                        queries_label_ids_val1 = []

                        for i in queries_label_ids_val.tolist():
                            temp = class_map_val[i]
                            queries_label_ids_val1.append(temp)

                        queries_label_ids_val1 = torch.LongTensor(queries_label_ids_val1)

                        support_input_ids_val = support_input_ids_val.to(device)
                        support_input_mask_val = support_input_mask_val.to(device)
                        support_segment_ids_val = support_segment_ids_val.to(device)
                        support_label_ids_val = support_label_ids_val.to(device)

                        queries_input_ids_val = queries_input_ids_val.to(device)
                        queries_input_mask_val = queries_input_mask_val.to(device)
                        queries_segment_ids_val = queries_segment_ids_val.to(device)
                        queries_label_ids_val1 = queries_label_ids_val1.to(device)

                        with torch.no_grad():

                            encoded_support_val, embs_support_val = model(support_input_ids_val, support_segment_ids_val, support_input_mask_val)
                            encoded_queries_val, embs_queries_val = model(queries_input_ids_val, queries_segment_ids_val, queries_input_mask_val)
                            
                            # encoded_support_val_cls = encoded_support_val[:,0,:]
                            # encoded_queries_val_cls = encoded_queries_val[:,0,:]
                            
                            encoded_support_val = torch.sum(encoded_support_val, dim =1)
                            support_mask_val = torch.sum(support_input_mask_val, dim =1).repeat(encoded_support_val.shape[1], 1).t().float()
                            encoded_support_val_avg = (encoded_support_val/support_mask_val)

                            encoded_queries_val = torch.sum(encoded_queries_val, dim =1)
                            queries_mask_val = torch.sum(queries_input_mask_val, dim =1).repeat(encoded_queries_val.shape[1], 1).t().float()
                            encoded_queries_val_avg = (encoded_queries_val/queries_mask_val)

                            # div2 = torch.norm(embs_support_val, dim =1)
                            div1 = torch.norm(encoded_queries_val_avg, dim =1)
                            # div1 = norm(encoded_queries_val_avg, axis=1).reshape(encoded_queries_val_avg.shape[0],1)
                
                            div2 = torch.norm(encoded_support_val_avg, dim =1)
                            # import ipdb; ipdb.set_trace()

                            res_val = encoded_queries_val_avg.mm(encoded_support_val_avg.t())/torch.mul(div1.unsqueeze(1), div2.unsqueeze(1).t())
                            # res1_val= torch.from_numpy(res1_val)
                            # res_val= res1_val.to(device)

                            s1_val = []
                            logits_support_max_val = torch.zeros([res_val.shape[0], args.N])
            
                            for i in range(args.N): 
                                d = (support_label_ids_val == class_batch_val[i]).nonzero().view(-1).tolist()
                                temp, _ = torch.max(res_val[:, d], dim = -1)
                                s1_val.append(temp)

                            logits_support_max_val = torch.stack(s1_val, dim =1) 

                        loss_fct = CrossEntropyLoss()
                        val_loss = loss_fct(logits_support_max_val.view(-1, args.N), queries_label_ids_val1)
                        total_val_loss +=val_loss
                        val_iter+=1
                    # val_result = compute_metrics(logits_support_max_val.detach().cpu().numpy(), queries_label_ids_val1.cpu().numpy())
                    # classes_batch = class_ids_val[(step*args.N):((step*args.N)+args.N)]
                        print("Validation loss", val_loss)
                        if len(preds_val) == 0:
                            preds_val.append(logits_support_max_val.detach().cpu().numpy())
                            ground_truth_val.append(queries_label_ids_val1.cpu().numpy())
                        else:
                            preds_val[0] = np.append(preds_val[0], logits_support_max_val.detach().cpu().numpy(), axis=0)
                            ground_truth_val[0] = np.append(ground_truth_val[0], queries_label_ids_val1.cpu().numpy(), axis =0)
                    # print(preds_val, ground_truth_val)
                    preds_val = np.argmax(preds_val[0], axis=1)
                    print("average validation loss", total_val_loss/val_iter)
                    print("validation accuracy", compute_metrics(preds_val, ground_truth_val[0]))
                ############################# Validation at each epoch#####################################################################

                #         print(name, param)
                # import ipdb; ipdb.set_trace()

            print("average train loss of epoch", avg_loss_epoch/num_iter_epoch)
            
            running_loss += (avg_loss_epoch/num_iter_epoch)
            count +=1
            print("Running loss", running_loss/count)

    #########################################################end of traning#############################################################            
    
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        support_examples, queries_examples, classes, labels = processor.get_dev_examples(args.data_dir, args.num_sets_dev, args.N, args.Q, args.K, 700)
        # for (ex_index, example) in enumerate(support_examples):
        #     print(example.label)
        # for (ex_index, example) in enumerate(queries_examples):
        #     print(example.label)
        support_features = convert_examples_to_features(support_examples, args.max_seq_length, tokenizer, labels)
        queries_features = convert_examples_to_features(queries_examples, args.max_seq_length, tokenizer, labels)
        # print(len(queries_features), (args.N * args.Q * args.num_sets_dev))
        assert len(queries_features) == (args.N * args.Q * args.num_sets_dev)

        label_map = {label : i for i, label in enumerate(labels)}
        class_ids = []
        for i in classes:
            class_id = label_map[i]
            class_ids.append(class_id)

        support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long)
        support_input_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long)
        support_segment_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long)
        support_label_ids = torch.tensor([f.label_id for f in support_features], dtype=torch.long)
        # print(support_label_ids)
        queries_input_ids = torch.tensor([f.input_ids for f in queries_features], dtype=torch.long)
        queries_input_mask = torch.tensor([f.input_mask for f in queries_features], dtype=torch.long)
        queries_segment_ids = torch.tensor([f.segment_ids for f in queries_features], dtype=torch.long)
        queries_label_ids = torch.tensor([f.label_id for f in queries_features], dtype=torch.long)  
        # print(queries_label_ids)
        support_data = TensorDataset(support_input_ids, support_input_mask, support_segment_ids, support_label_ids)
        queries_data = TensorDataset(queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids)

        support_dataloader = DataLoader(support_data, batch_size=support_batch_size)      
        queries_dataloader = DataLoader(queries_data, batch_size=queries_batch_size)

        model.eval()
        eval_loss = 0
        preds = []
        ground_truth = []
        nb_eval_steps = 0

        for step, batch in enumerate(zip(queries_dataloader, support_dataloader)):
            classes_batch = class_ids[(step*args.N):((step*args.N)+args.N)]
            queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids= batch[0][0], batch[0][1], batch[0][2], batch[0][3]
            support_input_ids, support_input_mask, support_segment_ids, support_label_ids= batch[1][0], batch[1][1], batch[1][2], batch[1][3]
            # print(classes_batch, queries_label_ids, len(queries_label_ids), support_input_ids, support_label_ids, len(support_label_ids))
            class_map = {label : i for i, label in enumerate(classes_batch)}
            # print(class_map)
            queries_label_ids1 = []
            for i in queries_label_ids.tolist():
                temp = class_map[i]
                queries_label_ids1.append(temp)

            queries_label_ids1 = torch.LongTensor(queries_label_ids1)
            queries_label_ids1 = queries_label_ids1.to(device)

            support_input_ids = support_input_ids.to(device)
            support_input_mask = support_input_mask.to(device)
            support_segment_ids = support_segment_ids.to(device)
            support_label_ids = support_label_ids.to(device)

            queries_input_ids = queries_input_ids.to(device)
            queries_input_mask = queries_input_mask.to(device)
            queries_segment_ids = queries_segment_ids.to(device)
            # queries_label_ids = queries_label_ids.to(device)
            with torch.no_grad():
                encoded_layers_support, embs_support = model(support_input_ids, support_segment_ids, support_input_mask)
                # print("embs support",embs_support)
                encoded_layers_queries, embs_queries = model(queries_input_ids, queries_segment_ids, queries_input_mask)
                # print("embs queries",embs_queries)
                # dims_support = list(logits_support.size())
                # logits_support_avg = torch.zeros([args.N, dims_support[1]])
                # logits_support_avg = logits_support_avg.to(device)
    
                # for i in range(len(classes_batch)):
                #     for j in range(dims_support[0]):
                #         if support_label_ids[j] == classes_batch[i]:
                #             logits_support_avg[i] = logits_support_avg[i] + logits_support[j]
                # # logits_support_avg[i] = logits_support_avg[i] / K
                # logits_support_avg = logits_support_avg/args.K
                # print("logits support average", logits_support_avg)
                
                # div2 = torch.norm(embs_support, dim=1)
                # # div1 = norm(encoded_layers_queries_avg, axis=1).reshape(encoded_layers_queries_avg.shape[0], 1)
                # div1 = torch.norm(embs_queries, dim =1)
                encoded_layer_support_cls = encoded_layers_support[:,0,:]
                encoded_layer_queries_cls = encoded_layers_queries[:,0,:]
                # encoded_layers_support = torch.sum(encoded_layers_support, dim =1)
                # support_mask = torch.sum(support_input_mask, dim =1).repeat(encoded_layers_support.shape[1], 1).t().float()
                # encoded_layers_support_avg = (encoded_layers_support/support_mask)
                
                # encoded_layers_queries = torch.sum(encoded_layers_queries, dim =1)
                # queries_mask = torch.sum(queries_input_mask, dim =1).repeat(encoded_layers_queries.shape[1], 1).t().float()
                # encoded_layers_queries_avg = (encoded_layers_queries/queries_mask)

                div2 = torch.norm(encoded_layer_support_cls, dim=1)
                # div1 = norm(encoded_layer_queries_cls, axis=1).reshape(encoded_layers_queries_avg.shape[0], 1)
                div1 = torch.norm(encoded_layer_queries_cls, dim =1)
                # div2 = norm(encoded_layer_support_cls, axis=1).reshape(encoded_layers_support_avg.shape[0], 1)

                res = encoded_layer_queries_cls.mm(encoded_layer_support_cls.t())/torch.mul(div1.unsqueeze(1), div2.unsqueeze(1).t())
                # print("res", res)
                s1 = []
                logits_support_max = torch.zeros([res.shape[0], args.N])

                for i in range(args.N): 
                    d = (support_label_ids == classes_batch[i]).nonzero().view(-1).tolist()
                    temp, _ = torch.max(res[:, d], dim = -1)
                    s1.append(temp)

                logits_support_max = torch.stack(s1, dim =1) 
                # import ipdb; ipdb.set_trace()
                # print("res",res)
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits_support_max.view(-1, args.N), queries_label_ids1)
            print("eval loss",tmp_eval_loss)               
            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits_support_max.detach().cpu().numpy())
                ground_truth.append(queries_label_ids1.cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits_support_max.detach().cpu().numpy(), axis=0)
                ground_truth[0] = np.append(ground_truth[0], queries_label_ids1.cpu().numpy(), axis =0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        # print(preds, ground_truth[0])
        result = compute_metrics(preds, ground_truth[0])
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['avg_eval_loss'] = eval_loss
        # result['global_step'] = global_step
        result['avg_train_loss'] = loss

        for key in sorted(result.keys()):
            print("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()

    




