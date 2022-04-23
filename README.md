# BERT NER on zero-shot-dev
## **This branch also contains a folder called roBERTa-zero-shot, which includes everything needed to run our second method - zero-shot with RoBERTa NER**

Use BERT-labels, created by J.C. Hunter (40204692) in the research of Zero Shot Learning for Named Entity Recognition on the CoNLL-2003 NER dataset.
For the original BERT model, revert to the main 'dev' branch.



# Requirements

-  `python3`
- `pip3 install -r requirements.txt`

# Run

Use the run_ner_script.ipynb, retrieve the code from the branch and download requirements using the cells on the script.
Run the below cell on the chosen python notebook to begin training of the researched model.

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_base_labels --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`

## The notebook also has options to download the trained model, and evaluate it if you insert it into the specified location (in my case, Google Drive)

# Result

## BERT-BASE
```
Accuracy: 0.9588515634081134
Recall: 0.9595207630568803
Precision: 0.23988019076422007
F1 score: 0.3838083052227521
```

## Pretrained model download from [here](https://1drv.ms/u/s!Auc3VRul9wo5hghurzE47bTRyUeR?e=08seO3)

## BERT-labels (researched method)
```
Accuracy: 0.21841218179704755
Recall: 0.2741864876696337
Precision: 0.05478134110787172
F1 score: 0.09131776541199914
```

## Pretrained model download from [here](https://drive.google.com/file/d/1NlxY6Bp4XHO02NChnsyX4FshrvpHTWrV/view?usp=sharing)

# Testing
## Testing is available in the final cell of the program. This will run tests in the code from bert.py, run_ner.py, and evaluate_model.py:
