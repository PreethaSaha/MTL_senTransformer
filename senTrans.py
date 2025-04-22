import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import random

# ----------- Model Definition -----------

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, num_classes_a=3, num_classes_b=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier_a = nn.Linear(hidden_dim, num_classes_a)  # Task A
        self.classifier_b = nn.Linear(hidden_dim, num_classes_b)  # Task B

    def forward(self, sentences):
        encoded = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits_a = self.classifier_a(cls_embedding)
        logits_b = self.classifier_b(cls_embedding)
        return logits_a, logits_b

# ----------- Dummy Dataset -----------

class DummyMultiTaskDataset(Dataset):
    def __init__(self, tokenizer, num_samples=40):
        self.sentences = [
            "The sky is blue.",
            "That was fantastic!",
            "Can you help me with this task?",
            "The cat sat on the mat.",
            "I love machine learning.",
            "This is terrible news.",
            "Please pass the salt.",
            "How are you today?",
            "I don't like this.",
            "She enjoys reading books."
        ]
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sentence = random.choice(self.sentences)
        label_a = torch.randint(0, 3, (1,)).item()
        label_b = torch.randint(0, 3, (1,)).item()
        return sentence, label_a, label_b

# ----------- Intents (Label Descriptions) -----------

task_a_intents = ['Statement', 'Question', 'Command']
task_b_intents = ['Positive', 'Neutral', 'Negative']

# ----------- Training Loop -----------

def train_model():
    model = MultiTaskSentenceTransformer()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    dataset = DummyMultiTaskDataset(model.tokenizer, num_samples=40)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model.train()
    num_epochs = 2

    for epoch in range(num_epochs):
        total_loss = 0
        correct_a, correct_b = 0, 0
        total_samples = 0

        for batch in loader:
            sentences, labels_a, labels_b = batch
            labels_a = torch.tensor(labels_a)
            labels_b = torch.tensor(labels_b)

            logits_a, logits_b = model(sentences)

            loss_a = loss_fn(logits_a, labels_a)
            loss_b = loss_fn(logits_b, labels_b)
            loss = loss_a + loss_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += len(sentences)

            preds_a = torch.argmax(logits_a, dim=1)
            preds_b = torch.argmax(logits_b, dim=1)
            correct_a += (preds_a == labels_a).sum().item()
            correct_b += (preds_b == labels_b).sum().item()

            # Print results per input
            for i in range(len(sentences)):
                print(f"\nInput: {sentences[i]}")
                print(f"  Task A - Predicted: {preds_a[i].item()} ({task_a_intents[preds_a[i]]}) | True: {labels_a[i].item()} ({task_a_intents[labels_a[i]]})")
                print(f"  Task B - Predicted: {preds_b[i].item()} ({task_b_intents[preds_b[i]]}) | True: {labels_b[i].item()} ({task_b_intents[labels_b[i]]})")

        acc_a = correct_a / total_samples
        acc_b = correct_b / total_samples
        avg_loss = total_loss / len(loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Task A Accuracy: {acc_a:.2f}")
        print(f"  Task B Accuracy: {acc_b:.2f}")

# ----------- Run -----------

if __name__ == "__main__":
    train_model()

