import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="macro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    # Read data
    # data = pd.read_csv("/content/drive/MyDrive/SupConst/datasets/archive/texts/train_titles.csv", names=["image_name", "description", "label"])

    # Define pretrained tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForSequenceClassification.from_pretrained(model_name, num_labels=101)

    # ----- 1. Preprocess data -----#
    # Preprocess data
    # X = list(data["description"])
    # data["label"] = pd.factorize(data.label)[0]
    # y = list(data["label"])
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    # X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    # train_dataset = Dataset(X_train_tokenized, y_train)
    # val_dataset = Dataset(X_val_tokenized, y_val)

    # # Define Trainer
    # args = TrainingArguments(
    #     output_dir="/content/drive/MyDrive/SupConst/output_test",
    #     evaluation_strategy="steps",
    #     eval_steps=500,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     num_train_epochs=3,
    #     seed=0,
    #     load_best_model_at_end=True,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # )

    # # Train pre-trained model
    # trainer.train()

    # ----- 3. Predict -----#
    # Load test data
    test_data = pd.read_csv("/content/drive/MyDrive/SupConst/datasets/archive/texts/test_titles.csv", names=["image_name", "description", "label"])
    X_test = list(test_data["description"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    # Load trained model
    model_path = "/content/drive/MyDrive/SupConst/output_test/checkpoint-5000/"
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=101)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    pd.DataFrame(raw_pred).to_csv("/content/drive/MyDrive/SupConst/output_test/predict_prob.csv")

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    print(y_pred)
