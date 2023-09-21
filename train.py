import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd


class TrainDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__()
        self.df = pd.read_csv(filename, sep="\t")
        self.encoding = tokenizer(self.df["text"].values.tolist(
        ), truncation=True, padding=True, return_tensors="pt")
        print("encoding : ")
        print(self.encoding)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = 1 if self.df["label"].iloc[index] == "MOS" else 0
        return {"input_ids": self.encoding["input_ids"][index], "attention_mask": self.encoding["attention_mask"][index], "label": label}


class TestDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super().__init__()
        self.df = pd.read_csv(filename, sep="\t")
        self.encoding = tokenizer(self.df["text"].values.tolist(
        ), truncation=True, padding=True, return_tensors="pt")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {"input_ids": self.encoding["input_ids"], "attention_mask": self.encoding["attention_mask"]}


model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds = TrainDataset("train.tsv", tokenizer=tokenizer)
validate_ds = TrainDataset("validate.tsv", tokenizer=tokenizer)
test_ds = TestDataset("test.tsv", tokenizer=tokenizer)

training_args = TrainingArguments(output_dir=model_name,
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  learning_rate=5e-5,
                                  weight_decay=0,
                                  max_grad_norm=1.0,
                                  num_train_epochs=10,
                                  lr_scheduler_type="linear",
                                  warmup_ratio=0,
                                  warmup_steps=0,
                                  log_level="passive",
                                  logging_strategy="epoch",
                                  fp16=True,
                                  run_name=model_name
                                  )

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_ds,
                  eval_dataset=validate_ds,
                  )

trainer.train()
