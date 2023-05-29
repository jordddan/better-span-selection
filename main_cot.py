import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
import random
import json
import argparse
import os, sys
import logging
import copy
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from seqeval.metrics import precision_score, recall_score, f1_score

import wandb

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    set_seed,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
)

from model import BertNegSampleForTokenClassification, RobertaNegSampleForTokenClassification

from data import *


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
  "bert": (BertConfig, BertNegSampleForTokenClassification, BertTokenizer),
  "roberta": (RobertaConfig, RobertaNegSampleForTokenClassification, RobertaTokenizer),
}

class Trainer:
    def __init__(self, 
                 args, 
                 model1, 
                 model2,
                 tokenizer, 
                 label_list, 
                 train_dataset=None, 
                 dev_dataset=None, 
                 test_dataset=None):
        self.args = args
        self.model1 = model1
        self.model2 = model2 
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.label_list = label_list
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.o_conf = None

        self.wandb_config = self.get_wandb_config()

        wandb.init(project=f"NER-{args.task}", config=self.wandb_config)

    def get_wandb_config(self):
        args = self.args
        args_dict = {}
        for arg in vars(args):
            args_dict[arg] = getattr(args, arg)
        
        return args_dict


    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=lambda x: x)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters1 = [
            {"params": [p for n, p in self.model1.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model1.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer_grouped_parameters2 = [
            {"params": [p for n, p in self.model2.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model2.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size (w. parallel, accumulation) = %d",
            self.args.batch_size * self.args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        
        best_score = 0.0
        global_steps = 0
        self.model1.zero_grad()
        self.model2.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            is_select = self.args.select_begin_epoch >= 0 and epoch >= self.args.select_begin_epoch
            if is_select:
                self.build_conf_mat()
                # self.god_eval()
            for step, batch_examples in enumerate(epoch_iterator):
                batch1 = self.convert_examples_to_features(batch_examples, training=True, is_select=is_select, flag=1)
                batch2 = self.convert_examples_to_features(batch_examples, training=True, is_select=is_select, flag=2)

                self.model1.train()
                self.model2.train()
                batch1 = {key: value.to(self.args.device) for key, value in batch1.items()}
                batch2 = {key: value.to(self.args.device) for key, value in batch2.items()}

                loss1 = self.model2(**batch1).loss
                loss2 = self.model1(**batch2).loss
                if self.args.gradient_accumulation_steps > 1:
                    loss1 = loss1 / self.args.gradient_accumulation_steps
                    loss2 = loss2 / self.args.gradient_accumulation_steps
                loss1.backward()
                loss2.backward()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    torch.nn.utils.clip_grad_norm_(self.model1.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.args.max_grad_norm)
                    scheduler1.step()  # Update learning rate schedule
                    optimizer1.step()
                    scheduler2.step()
                    optimizer2.step()
                    self.model1.zero_grad()
                    self.model2.zero_grad()
                    
                    if global_steps % self.args.save_steps == 0:
                        result, _ = self.evaluate(mode="dev", prefix=global_steps)
                        logger.info(", ".join("%s: %s" % item for item in result.items()))
                        wandb.log({"f1-dev":result["f1"]})
                        if self.args.eval_test_set:
                            result, _ = self.evaluate(mode="test", prefix="test")
                            wandb.log({"f1-test":result["f1"]})
                            logger.info(", ".join("%s: %s" % item for item in result.items()))
                        if result["f1"] > best_score:
                            logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
                            best_score = result["f1"]
                            # Save the best model checkpoint
                            output_dir = os.path.join(self.args.output_dir, "checkpoint-best")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            
                            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving the best model checkpoint to %s", output_dir)
                            
        # logger.info("Saving model checkpoint to %s", self.args.output_dir)
        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(self.args.output_dir)
        # self.tokenizer.save_pretrained(self.args.output_dir)

    def convert_examples_to_features(self, examples, training=False, is_select=False, flag = 1):
        batch_input_ids, batch_input_mask, batch_start_pos = [], [], []
        label_map = {label: i for i, label in enumerate(self.label_list)}
        sentence_length = [len(example.words) for example in examples]
        max_sentence_length = max(sentence_length)

        for ex_index, example in enumerate(examples):
            tokens, subword_len = [], []
            for word in example.words:
                word_tokens = self.tokenizer.tokenize(word)
                if len(word) != 0 and len(word_tokens) == 0:
                    word_tokens = [self.tokenizer.unk_token]
                tokens.extend(word_tokens)
                subword_len.append(len(word_tokens))
                
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            if self.args.model_type == "roberta":
                tokens += [self.tokenizer.sep_token]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            start_pos = np.cumsum([1] + subword_len[:-1]).tolist()

            batch_input_ids.append(input_ids)
            batch_input_mask.append(input_mask)
            batch_start_pos.append(start_pos)
        
        # padding
        max_seq_length = max([len(tokens) for tokens in batch_input_ids])
        for input_ids, input_mask, start_pos in zip(batch_input_ids, batch_input_mask, batch_start_pos):
            padding_length = max_seq_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            input_mask += [0] * padding_length
            start_pos += [0] * (max_sentence_length - len(start_pos))
            
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(batch_input_mask, dtype=torch.long)
        batch_start_pos = torch.tensor(batch_start_pos, dtype=torch.long)
        eval_batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_input_mask,
            "labels_mat": None,
            "start_pos": batch_start_pos,
        }
        if not training:
            return eval_batch
        
        batch_positive_samples, batch_sample_candidates_correct, batch_sample_candidates_potential = self.posnegsample_select(flag, examples, is_select)
        
        batch_label_spans = []
        for ex_index, (positive_samples, sample_candidates_c, sample_candidates_p) in enumerate(zip(batch_positive_samples, batch_sample_candidates_correct, batch_sample_candidates_potential)):
            label_span_mat = np.full((max_sentence_length, max_sentence_length), self.pad_token_label_id, dtype=int)
            for span_b, span_e, span_label_id in positive_samples:
                label_span_mat[span_b, span_e] = span_label_id
                
            sample_candidates = sample_candidates_c if self.args.select_negative and not is_select else sample_candidates_c + sample_candidates_p
            neg_num = int(sentence_length[ex_index] * self.args.neg_sample_rate) + 1
            if len(sample_candidates) > 0:
                sample_num = min(neg_num, len(sample_candidates))
                np.random.shuffle(sample_candidates)
                for span_b, span_e in sample_candidates[:sample_num]:
                    label_span_mat[span_b, span_e] = label_map["O"]
                
            batch_label_spans.append(label_span_mat.tolist())

        batch_label_spans = torch.tensor(batch_label_spans, dtype=torch.long)

        train_batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_input_mask,
            "labels_mat": batch_label_spans,
            "start_pos": batch_start_pos,
        }
        return train_batch


    def evaluate(self, mode, prefix=""):
        if mode == "dev":
            eval_dataset = self.dev_dataset
        elif mode == "test":
            eval_dataset = self.test_dataset
            
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=lambda x: x)
        # Eval!
        logger.info("***** Running evaluation %s - %s *****" % (mode, prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        all_pred_spans = []
        self.model1.eval()
        for batch_examples in tqdm(eval_dataloader, desc="Evaluating"):
            batch = self.convert_examples_to_features(batch_examples)
            batch = {key: value.to(self.args.device) for key, value in batch.items() if value is not None}
            
            with torch.no_grad():
                batch_logits_mat1 = self.model1(**batch).logits
                batch_logits_mat2 = self.model2(**batch).logits

            batch_logits_mat = (batch_logits_mat1.detach() + batch_logits_mat2.detach()) / 2
            
            for example, logits_mat in zip(batch_examples, batch_logits_mat):
                pred_spans = self.convert_logits_to_preds(logits_mat, example.length)
                all_pred_spans.append(pred_spans)

        all_sentences, all_bio_preds, all_bio_labels = [], [], []
        for pred_spans, example in zip(all_pred_spans, eval_dataset):
            bio_preds = bio_tagging(pred_spans, example.length)
            bio_labels = example.bio_labels
            assert len(bio_preds) == len(bio_labels)
            all_sentences.append(example.words)
            all_bio_preds.append(bio_preds)
            all_bio_labels.append(bio_labels)
        
        results = {
            "f1": f1_score(all_bio_labels, all_bio_preds),
            "precision": precision_score(all_bio_labels, all_bio_preds),
            "recall": recall_score(all_bio_labels, all_bio_preds),
        }
        return results, all_bio_preds

    def convert_logits_to_preds(self, logits_mat, length):
        conf_table, index_table = torch.max(logits_mat, dim=-1)
        conf_table = conf_table.cpu().numpy()
        index_table = index_table.cpu().numpy()
        candidate_spans = []
        label_map = {i: label for i, label in enumerate(self.label_list)}
        for i in range(length):
            for j in range(i, length):
                label_id = index_table[i, j]
                if label_id != self.pad_token_label_id and label_map[label_id] != "O":
                    candidate_spans.append(Span(l=i, r=j, label=label_map[label_id], conf=conf_table[i, j]))

        candidate_spans.sort(key=lambda x: x.conf, reverse=True)
        final_spans = []
        for span in candidate_spans:
            if not conflict_judge(final_spans, span):
                final_spans.append(span)
        return final_spans
    
    def posnegsample_select(self, flag, examples, is_selct):
        label_map = {label: i for i, label in enumerate(self.label_list)}
        batch_size = len(examples)
        positive_samples = [[] for _ in range(batch_size)]
        # sample_candidates = [[] for _ in range(batch_size)]
        sample_candidates_correct = [[] for _ in range(batch_size)]
        sample_candidates_potential = [[] for _ in range(batch_size)]
        for ex_index, example in enumerate(examples):
            label_span_mat = np.zeros((example.length, example.length), dtype=int)
            # 0: correct O, 1: potenial O, 2: label
            
            for span_b, span_e, span_label in example.label_spans:
                label_span_mat[span_b, span_e] = 2
                if flag == 1: 
                    mat = example.conf_mat1
                else:
                    mat = example.conf_mat2
                if not is_selct or mat[span_b, span_e]:
                    positive_samples[ex_index].append((span_b, span_e, label_map[span_label]))
                    continue
            tmp_span_list = [(-1, -1, -1)] + example.label_spans + [(example.length, example.length, -1)]
            for span_id in range(1, len(tmp_span_list)):
                empty_l = tmp_span_list[span_id - 1][1] + 1
                empty_r = tmp_span_list[span_id][0] - 1
                if empty_l > empty_r:
                    continue
                for l in range(empty_l, empty_r + 1):
                    for r in range(l, empty_r + 1):
                        label_span_mat[l, r] = 1
            for l in range(example.length):
                for r in range(l, example.length):
                    if label_span_mat[l, r] == 2:
                        continue
                    if label_span_mat[l, r] == 0:
                        sample_candidates_correct[ex_index].append((l, r))
                    else:
                        if flag == 1:
                            mat = example.conf_mat1
                        else:
                            mat = example.conf_mat2
                        if not is_selct or mat[l, r]:
                            sample_candidates_potential[ex_index].append((l, r))
        return positive_samples, sample_candidates_correct, sample_candidates_potential
        
    def build_conf_mat(self):
        dataset = self.train_dataset
        label_map = {label: i for i, label in enumerate(self.label_list)}

        eval_dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=lambda x: x)
        logger.info("***** Build Conf Mat *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        
        all_logits_mat1 = []
        all_logits_mat2 = []
        self.model1.eval()
        self.model2.eval()
        for batch_examples in tqdm(eval_dataloader, desc="Evaluating"):
            batch = self.convert_examples_to_features(batch_examples)
            batch = {key: value.to(self.args.device) for key, value in batch.items() if value is not None}
            with torch.no_grad():
                batch_logits_mat1 = self.model1(**batch).logits
                batch_logits_mat2 = self.model2(**batch).logits
            batch_logits_mat1 = batch_logits_mat1.detach().cpu()
            batch_logits_mat2 = batch_logits_mat2.detach().cpu()
            for example, logits_mat in zip(batch_examples, batch_logits_mat1):
                all_logits_mat1.append(logits_mat[:example.length, :example.length])
            for example, logits_mat in zip(batch_examples, batch_logits_mat2):
                all_logits_mat2.append(logits_mat[:example.length, :example.length])
                
        # conf mat init
        for example in dataset:
            example.conf_mat1 = np.ones((example.length, example.length), dtype=bool)
            example.conf_mat2 = np.ones((example.length, example.length), dtype=bool)
        
        
        if self.args.select_positive:
            conf_list1 = []
            label_list1 = []
            pred_list1 = []
            conf_list2 = []
            label_list2 = []
            pred_list2 = []

            for ex_index, (example, logits_mat) in enumerate(zip(dataset, all_logits_mat1)):
                for span_b, span_e, span_label in example.label_spans:
                    label_id = label_map[span_label]
                    entity_conf = logits_mat[span_b, span_e].softmax(dim=-1)[label_id].item()
                    pred_label = logits_mat[span_b, span_e].argmax(dim=-1).item()
                    conf_list1.append(entity_conf)
                    label_list1.append(label_map[span_label])
                    pred_list1.append(pred_label)

            for ex_index, (example, logits_mat) in enumerate(zip(dataset, all_logits_mat2)):
                for span_b, span_e, span_label in example.label_spans:
                    label_id = label_map[span_label]
                    entity_conf = logits_mat[span_b, span_e].softmax(dim=-1)[label_id].item()
                    pred_label = logits_mat[span_b, span_e].argmax(dim=-1).item()
                    conf_list2.append(entity_conf)
                    label_list2.append(label_map[span_label])
                    pred_list2.append(pred_label)

            conf_list1 = np.array(conf_list1)
            label_list1 = np.array(label_list1)
            pred_list1 = np.array(pred_list1)

            conf_list2 = np.array(conf_list2)
            label_list2 = np.array(label_list2)
            pred_list2 = np.array(pred_list2)

            entity_mean1 = []
            for label in self.label_list:
                if label == "O":
                    entity_mean1.append(0)
                    continue
                label_id = label_map[label]
                r = ((label_list1 == label_id) & (pred_list1 == label_id)).sum() / (label_list1 == label_id).sum()
                entity_mean1.append(conf_list1[label_list1 == label_id].mean() if r > self.args.pbegin_f1 else 0)
            entity_mean2 = []
            for label in self.label_list:
                if label == "O":
                    entity_mean2.append(0)
                    continue
                label_id = label_map[label]
                r = ((label_list2 == label_id) & (pred_list2 == label_id)).sum() / (label_list2 == label_id).sum()
                entity_mean2.append(conf_list2[label_list2 == label_id].mean() if r > self.args.pbegin_f1 else 0)

            logger.info("entity conf1: %s" % str(entity_mean2))

                
            for ex_index, (example, logits_mat) in enumerate(zip(dataset, all_logits_mat1)):
                for span_b, span_e, span_label in example.label_spans:
                    label_id = label_map[span_label]
                    entity_conf = logits_mat[span_b, span_e].softmax(dim=-1)[label_id].item()
                    if entity_conf < entity_mean1[label_id]:
                        example.conf_mat1[span_b, span_e] = False

            for ex_index, (example, logits_mat) in enumerate(zip(dataset, all_logits_mat2)):
                for span_b, span_e, span_label in example.label_spans:
                    label_id = label_map[span_label]
                    entity_conf = logits_mat[span_b, span_e].softmax(dim=-1)[label_id].item()
                    if entity_conf < entity_mean1[label_id]:
                        example.conf_mat2[span_b, span_e] = False
        
        if self.args.select_negative:
            for example, logits in zip(dataset, all_logits_mat1):
                tmp_span_list = [(-1, -1, -1)] + example.label_spans + [(example.length, example.length, -1)]
                for span_id in range(1, len(tmp_span_list)):
                    empty_l = tmp_span_list[span_id - 1][1] + 1
                    empty_r = tmp_span_list[span_id][0] - 1
                    if empty_l > empty_r:
                        continue
                    for l in range(empty_l, empty_r + 1):
                        for r in range(l, empty_r + 1):
                            if logits[l, r].argmax().item() != label_map["O"]:
                                example.conf_mat1[l, r] = False

        if self.args.select_negative:
            for example, logits in zip(dataset, all_logits_mat2):
                tmp_span_list = [(-1, -1, -1)] + example.label_spans + [(example.length, example.length, -1)]
                for span_id in range(1, len(tmp_span_list)):
                    empty_l = tmp_span_list[span_id - 1][1] + 1
                    empty_r = tmp_span_list[span_id][0] - 1
                    if empty_l > empty_r:
                        continue
                    for l in range(empty_l, empty_r + 1):
                        for r in range(l, empty_r + 1):
                            if logits[l, r].argmax().item() != label_map["O"]:
                                example.conf_mat2[l, r] = False

def main():
    parser = argparse.ArgumentParser()
  
    ## Required parameters
    parser.add_argument("--data_dir", default="./data", type=str, required=True)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--log_file", default="train.log", type=str)
    parser.add_argument("--task", default="conll", type=str)

    parser.add_argument("--labels", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test_set", action="store_true", help="Whether to run prediction on the test set during training.")
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run predictions on the test set")

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--mlp_hidden_size", default=256, type=int)
    parser.add_argument("--mlp_dropout_rate", default=0.2, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=20, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    
    parser.add_argument("--neg_sample_rate", type=float, default=0.35)
    parser.add_argument("--select_begin_epoch", type=int, default=-1)
    parser.add_argument("--pbegin_f1", type=float, default=0.5)
    parser.add_argument("--select_positive", action="store_true")
    parser.add_argument("--select_negative", action="store_true")
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Setup logging
    logging.basicConfig(handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logging.info("Input args: %r" % args)
    
    # Set seed
    set_seed(args.seed)
    
    labels = get_labels(args.labels)
    num_labels = len(labels)
    
    # load cls model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    mlp_config = {"hidden_size": args.mlp_hidden_size,
                  "dropout_rate": args.mlp_dropout_rate}
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    
    if args.do_train:
        train_dataset = load_examples(args, mode="train")
        dev_dataset = load_examples(args, mode="dev")
        test_dataset = load_examples(args, mode="test")
        
        model1 = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            mlp_config=mlp_config,
        ).to(args.device)
        
        model2 = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            mlp_config=mlp_config,
        ).to(args.device)

        trainer = Trainer(
            args=args,
            model1=model1,
            model2=model2,
            tokenizer=tokenizer,
            label_list=labels,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
        )
        trainer.train()
        
    if args.do_eval:
        model_path = os.path.join(args.output_dir if args.do_train else args.model_name_or_path, "checkpoint-best")
        
        model = model_class.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            mlp_config=mlp_config,
        ).to(args.device)
        
        test_dataset = load_examples(args, mode="test")
        trainer = Trainer(
            args=args,
            model=model,
            tokenizer=tokenizer,
            label_list=labels,
            test_dataset=test_dataset,
        )
        result, predictions = trainer.evaluate(mode="test")
        logger.info(", ".join("%s: %s" % item for item in result.items()))
    
    
if __name__ == "__main__":
    main()