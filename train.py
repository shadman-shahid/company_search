import numpy as np
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,AutoConfig
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import EvalPrediction
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
nltk.download('punkt')

data = []
with open('./data/final_dataset.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
print(data[0])

def compute_metrics(eval_pred: EvalPrediction):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    num_labels = eval_pred.predictions.shape[-1]

    precision, recall, f1, support = [], [], [], []
    for label in range(num_labels):
        label_preds = preds[:, label]
        label_labels = labels[:, label]
        p, r, f, s = precision_recall_fscore_support(label_labels, label_preds, average='weighted')
        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(s)

    precision_macro = sum(precision) / len(precision)
    recall_macro = sum(recall) / len(recall)
    f1_macro = sum(f1) / len(f1)

    return {
        # 'eval_loss': eval_pred.loss,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_precision_macro': precision_macro,
        'eval_recall_macro': recall_macro,
        'eval_f1_macro': f1_macro
    }

def get_labels(text, labels):
    token_labels = []
    tokens = word_tokenize(text)
    idx = 0
    for token in tokens:
        label = 'O'  # Default label
        for entity in labels:
            start = entity['start_offset']
            end = entity['end_offset']
            lbl = entity['label']
            if idx >= start and idx < end:
                label = lbl
                break
        token_labels.append(label)
        idx += len(token) + 1  # +1 for the space after each token
    return tokens, token_labels


def convert_data_to_ner_format(data):
    sentences = []
    label_lists = []
    for item in data:
        text = item['text']
        labels = item.get('entities', [])
        sents = sent_tokenize(text)
        for sent in sents:
            tokens, token_labels = get_labels(sent, labels)
            sentences.append(tokens)
            label_lists.append(token_labels)
    return sentences, label_lists


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].to(device).cpu() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


if __name__=='__main__':
    sentences, label_lists = convert_data_to_ner_format(data)

    unique_labels = list(set(label for labels in label_lists for label in labels))
    unique_labels = sorted(unique_labels)
    label_to_id = {label: id for id, label in enumerate(unique_labels)}
    id_to_label = {id: label for label, id in label_to_id.items()}
    labels_numerical = [[label_to_id[label] for label in labels] for labels in label_lists]

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    encoded_inputs = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)

    config = AutoConfig.from_pretrained(model_checkpoint)
    config.num_labels = len(unique_labels)
    config.id2label = id_to_label
    config.label2id = label_to_id

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config)
    # model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels = len(unique_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels = []
    for doc_labels, doc_offset in zip(labels_numerical, encoded_inputs["offset_mapping"]):
        doc_enc_labels = []
        curr_pos = 0
        for offset in doc_offset:
            if offset[0] == 0 and offset[1] != 0:
                doc_enc_labels.append(doc_labels[curr_pos])
                curr_pos += 1
            else:
                doc_enc_labels.append(-100)  # -100 is the index that is ignored by PyTorch's Cross Entropy Loss
        labels.append(doc_enc_labels)
    labels = torch.tensor(labels).to(device).cpu()
    train_dataset = NERDataset(encoded_inputs, labels[:int(0.8 * len(labels))])
    test_dataset = NERDataset(encoded_inputs, labels[int(0.8 * len(labels)):])

    training_args = TrainingArguments(
        output_dir="ner_output",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="ner_logs",
        logging_steps=200,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
