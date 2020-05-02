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
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)

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
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            print("lines",lines)
            return lines

def intialise(input_file, n, q, k, class_size):
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
        # self.labels = labels
        # print("labels", labels)

        classes = random.sample(labels, n)
        print("Classes", classes)
        queries = []
        support = []
        for i in classes:
            j = labels.index(i)
            query = np.random.choice(list(range(j*class_size, (j+1)*class_size)), q+k, False)
            queries = queries + query.tolist()[:q]
            support = support + query.tolist()[q:]
        lines = np.asarray(lines)
        lines_subset1 = lines[support]
        support = lines_subset1.tolist()
        #print("Support",support)
        lines_subset2 = lines[queries]
        queries = lines_subset2.tolist()
        #print("Queries",queries)
        return labels, classes, support, queries

class BertForRelationClassification(BertPreTrainedModel):

    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertForRelationClassification, self).__init__(config)
        self.output_attentions = output_attentions
        # self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        _, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        return pooled_output

class MultiClassClassification(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def __init__(self, input_file, n, q, k, class_size):
        self.labels, self.classes, self.support, self.queries = intialise(input_file, n, q, k, class_size)

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_full.tsv")))
        return self._create_examples(self.support, "train"), self._create_examples(self.queries, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.support, "dev"), self._create_examples(self.queries, "dev")

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
        
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

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
    assert len(preds) == len(labels)
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
                        default=128,
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
    parser.add_argument("--queries_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
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

    args.queries_batch_size = args.queries_batch_size // args.gradient_accumulation_steps

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

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForRelationClassification.from_pretrained(args.bert_model, cache_dir=cache_dir)
    
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    num_sets_train = 3              ################################################
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    N= 5
    Q1 = 100
    K = 5
    class_size = 700
    for i in range(num_sets_train):

        logger.info("################### TRAIN STEP %d ############################", i)    

        processor = MultiClassClassification(os.path.join(args.data_dir, "dev_full.tsv"), N, Q1, K, class_size)      
        label_list = processor.get_labels()
        num_labels = len(label_list)

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        if args.do_train:

            # Prepare data loader

            train_support, train_queries = processor.get_train_examples(args.data_dir)
            support_features = convert_examples_to_features(train_support, args.max_seq_length, tokenizer, label_list)
            support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long)
            support_input_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long)
            support_segment_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long)
            support_label_ids = torch.tensor([f.label_id for f in support_features], dtype=torch.long)
            
            queries_features = convert_examples_to_features(train_queries, args.max_seq_length, tokenizer, label_list)
            queries_input_ids = torch.tensor([f.input_ids for f in queries_features], dtype=torch.long)
            queries_input_mask = torch.tensor([f.input_mask for f in queries_features], dtype=torch.long)
            queries_segment_ids = torch.tensor([f.segment_ids for f in queries_features], dtype=torch.long)
            queries_label_ids = torch.tensor([f.label_id for f in queries_features], dtype=torch.long)  

            # support_data = TensorDataset(support_input_ids, support_input_mask, support_segment_ids, support_label_ids)
            queries_data = TensorDataset(queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids)

            support_input_ids = support_input_ids.to(device)
            support_input_mask = support_input_mask.to(device)
            support_segment_ids = support_segment_ids.to(device)
            support_label_ids = support_label_ids.to(device)

            # queries_input_ids = queries_input_ids.to(device)
            # queries_input_mask = queries_input_mask.to(device)
            # queries_segment_ids = queries_segment_ids.to(device)
            # queries_label_ids = queries_label_ids.to(device)

            if args.local_rank == -1:
                # support_sampler = RandomSampler(support_data)
                queries_sampler = RandomSampler(queries_data)
            else:
                # support_sampler = DistributedSampler(support_data)
                queries_sampler = DistributedSampler(queries_data)
            # support_dataloader = DataLoader(support_data, sampler=support_sampler, batch_size=args.support_batch_size)      
            queries_dataloader = DataLoader(queries_data, sampler=queries_sampler, batch_size=args.queries_batch_size)      

            num_train_optimization_steps = len(queries_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
            if args.local_rank != -1:
                num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

            optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_queries))
            logger.info("  Batch size = %d", args.queries_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)


            model.train()
            logits_support = model(support_input_ids, support_segment_ids, support_input_mask)
            dims_support = list(logits_support.size())
            logits_support_avg = torch.zeros([num_labels, dims_support[1]])
            logits_support_avg = logits_support_avg.to(device)
            # sum_support = torch.zeros([num_labels, dims_support[1]])
            # sum_support = sum_support.to(device)
            # print("support_label",support_label_ids)
            for i in range(num_labels):
                for j in range(dims_support[0]):
                    if support_label_ids[j] == i:
                        logits_support_avg[i] = logits_support_avg[i] + logits_support[j]
                # logits_support_avg[i] = logits_support_avg[i] / K
            logits_support_avg = logits_support_avg/K
            # print("logits_support",logits_support_avg, logits_support_avg.size())            

            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                # tr_loss = 0
                # nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(queries_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids= batch

                    logits_queries = model(queries_input_ids, queries_segment_ids, queries_input_mask)
                    # print("logits_queries",logits_queries, logits_queries.size())            

                    res = logits_queries.mm(logits_support_avg.t())
                    # temp = torch.nn.functional.softmax(res, dim=1)
                    print("train res",res.view(-1, num_labels), queries_label_ids.view(-1))

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(res.view(-1, num_labels), queries_label_ids.view(-1))
                    print("train loss",loss)

                    if n_gpu > 1:
                        loss = loss.mean() 
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward(retain_graph=True)
                    # optimizer.step()
                    # optimizer.zero_grad()

                    tr_loss += loss.item()
                    # nb_tr_examples += queries_input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        # global_step += 1


    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForRelationClassification.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForRelationClassification.from_pretrained(args.bert_model)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        num_sets_val = 100            ################################################
        global_accuracy = 0
        Q2 = 100

        for i in range(num_sets_val):

            logger.info("################### EVAL STEP %d ############################", i)    

            processor = MultiClassClassification(os.path.join(args.data_dir, "dev_full.tsv"), N, Q2, K, class_size)    
            label_list = processor.get_labels()

            eval_support, eval_queries = processor.get_dev_examples(args.data_dir)
            support_features = convert_examples_to_features(eval_support, args.max_seq_length, tokenizer, label_list)
            support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long)
            support_input_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long)
            support_segment_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long)
            support_label_ids = torch.tensor([f.label_id for f in support_features], dtype=torch.long)
            
            queries_features = convert_examples_to_features(eval_queries, args.max_seq_length, tokenizer, label_list)
            queries_input_ids = torch.tensor([f.input_ids for f in queries_features], dtype=torch.long)
            queries_input_mask = torch.tensor([f.input_mask for f in queries_features], dtype=torch.long)
            queries_segment_ids = torch.tensor([f.segment_ids for f in queries_features], dtype=torch.long)
            queries_label_ids = torch.tensor([f.label_id for f in queries_features], dtype=torch.long)  

            # support_data = TensorDataset(support_input_ids, support_input_mask, support_segment_ids, support_label_ids)
            queries_data = TensorDataset(queries_input_ids, queries_input_mask, queries_segment_ids, queries_label_ids)

            support_input_ids = support_input_ids.to(device)
            support_input_mask = support_input_mask.to(device)
            support_segment_ids = support_segment_ids.to(device)
            support_label_ids = support_label_ids.to(device)

            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_queries))
            logger.info("  Batch size = %d", args.eval_batch_size)

            eval_sampler = SequentialSampler(queries_data)
            queries_dataloader = DataLoader(queries_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []
            # print("support features",support_input_ids, support_input_mask, support_segment_ids)
            logits_support = model(support_input_ids, support_segment_ids, support_input_mask)
            # print("logits_support", logits_support)
            dims_support = list(logits_support.size())
            logits_support_avg = torch.zeros([num_labels, dims_support[1]])
            logits_support_avg = logits_support_avg.to(device)
            # sum_support = torch.zeros([num_labels, dims_support[1]])
            # sum_support = sum_support.to(device)
            # print(support_label_ids)

            for i in range(num_labels):
                for j in range(dims_support[0]):
                    if support_label_ids[j] == i:
                        logits_support_avg[i] = logits_support_avg[i] + logits_support[j]
            logits_support_avg = logits_support_avg/K
            
                # logits_support_avg[i] = logits_support_avg[i] / K
            # print("logits_support_avg",logits_support_avg)
            
            for input_ids, input_mask, segment_ids, label_ids in tqdm(queries_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                # print("query features", input_ids, input_mask, segment_ids)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)
                    # print("logits_queries", logits)
                    res = logits.mm(logits_support_avg.t())
                    print("eval res", res)
                    # print(label_ids)
                # create eval loss and other metric required by the task
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(res.view(-1, num_labels), label_ids.view(-1))
                print("eval loss",tmp_eval_loss)               
                eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(res.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], res.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            preds = np.argmax(preds, axis=1)
            
            result = compute_metrics(preds, queries_label_ids.numpy())
            loss = tr_loss/nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            global_accuracy += result['acc']

        logger.info("Global Accuracy: %f", global_accuracy/num_sets_val)

if __name__ == "__main__":
    main()
                

