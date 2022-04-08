"""BERT NER Inference."""

from __future__ import absolute_import, division, print_function

import json
import os

import torch
import torch.nn.functional as F
from nltk import word_tokenize
from pytorch_transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)


class BertNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits

class Ner:

    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        model = BertNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        return output
      
    def predict_original(self, text: list, ground_truth: list):
      
        entities_selected = 0
        entities_relevant = 0
        true_positives = 0
 
        input = ' '.join(text)
        print(input)
    
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(input)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
  
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]
        
        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(input)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]

        simplified_labels = { "O": "O", "B-MISC": "miscellaneous", "I-MISC": "miscellaneous", "B-PER": "person", "I-PER": "person", 
                         "B-ORG": "organisation", "I-ORG": "organisation", "B-LOC": "location", "I-LOC": "location", "[CLS]": "[CLS]", "[SEP]": "[SEP]" } 
        
        for i, label in enumerate(output):
            output[i]["tag"] = simplified_labels[label["tag"]]
        
        # make dict for text and ground truth
        # using dictionary comprehension
        # to convert lists to dictionary
        result_dict = {text[i]: ground_truth[i] for i in range(len(text))}
        
        labels = ["person", "organisation", "location"]
        
        # make groups of words that model finds similar
        for label in labels:
            print("|------------------------------------------------------|")
            print("| Model groups these words to be common with: " + str(label) + " |")
            print("|------------------------------------------------------|")
            for predicted in output:
              selected = result_dict.get(predicted["word"])
              print(selected)
              if selected:
                if predicted["tag"] != 'O':
                    entities_selected += 1
                    if predicted["tag"] == label:
                      #print(predicted_label["word"])
                      entities_relevant += 1
                      if selected == label:
                          print("correct")
                          true_positives += 1
                          #else: print("incorrect")
                          #else: print("incorrect")
            print("|------------------------------------------------------|")
        
        return true_positives, entities_selected, entities_relevant
      
    def predict_zero_shot(self, text: list, label_list: list, ground_truth: list):
      
        entities_selected = 0
        entities_relevant = 0
        true_positives = 0
      
        input = ' '.join(text) + " [SEP] " + ' '.join(label_list)
        print(input)
        print(len(input.split()))
        input_ids,input_mask,segment_ids,valid_ids = self.preprocess(input)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
  
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]
        
        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(input)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]

        # make dict for text and ground truth
        # using dictionary comprehension
        # to convert lists to dictionary
        result_dict = {text[i]: ground_truth[i] for i in range(len(text))}
        
        # make groups of words that model finds similar
        # for amount of labels (labels after sep) make a section that prints all words with that label
        sep_pos = words.index("SEP") # need to find position of seperator
        print("sep pos: " + str(sep_pos))
        
        before_sep = output[:sep_pos-1]
        after_sep = output[sep_pos+2:len(input)]
        
        for determined_label in after_sep:
            print("|------------------------------------------------------|")
            print("| Model groups these words to be common with: " + str(determined_label["word"]) + " |")
            print("|------------------------------------------------------|")
            for predicted_label in before_sep:
              if predicted_label["tag"] != 'O':
                  entities_selected += 1
                  if predicted_label["tag"] is determined_label["tag"]:
                      entities_relevant += 1
                      if determined_label["word"] == result_dict[str(predicted_label["word"])]:
                        print("correct")
                        true_positives += 1
                      else: print("incorrect")
                  else: print("incorrect")
            print("|------------------------------------------------------|")
        
        return true_positives, entities_selected, entities_relevant

