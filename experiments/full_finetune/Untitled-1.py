
import warnings
warnings.simplefilter("ignore")


from datasets import load_dataset

import numpy as np

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, AutoConfig, get_scheduler

from sklearn.metrics import accuracy_score

model_name = "prajjwal1/bert-tiny"
EPOCHS = 4
BATCH_SIZE = 32
LR = 2e-5


# Prepare data
dataset = load_dataset("glue", "sst2")
num_labels = dataset["train"].features["label"].num_classes


tokenizer = AutoTokenizer.from_pretrained(model_name)


tokenized_dataset = dataset.map(
    lambda example: tokenizer(example["sentence"], max_length=128, padding='max_length', truncation=True),
    batched=True
)


tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

tokenized_dataset = tokenized_dataset.remove_columns(['idx'])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")


train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=False, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(tokenized_dataset["validation"], shuffle=False, batch_size=BATCH_SIZE)


EPSILON = 8.0
DELTA = 1e-6
MAX_GRAD_NORM = 0.01


config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


model = model.train()


optimizer = SGD(model.parameters(), lr=LR)

num_training_steps = EPOCHS * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable Parameters: {trainable_params} || All Parameters: {all_param} || Trainable Parameters (%): {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(model)


def train(model, train_dataloader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    epsilon = []

    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch: {epoch}"):
        
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        if i % 8000 == 0:

            print(f"Training Epoch: {epoch} | Loss: {np.mean(losses):.6f}")                    


for tqdm_epoch in tqdm(range(1, EPOCHS+1), desc="Training"):
    train(model, train_dataloader, optimizer, EPOCHS, device)