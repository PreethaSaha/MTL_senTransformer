
import torch
import torch.nn as nn
import torch.optim as optim
import nbimporter
from task2 import MultiTaskSentenceTransformer

# Hypothetical Data (for illustration only)
sample_sentences = [
    "AI is reshaping industries.",        # Tech, Positive
    "The game ended in a draw.",          # Sports, Negative
    "It rained heavily last night.",      # Weather, Negative
    "This new phone is amazing.",         # Tech, Positive
    "The forecast predicts thunderstorms." # Weather, Negative
]

# Labels: Task A = Topic, Task B = Sentiment
labels_a = torch.tensor([0, 1, 2, 0, 2])  # 0=Tech, 1=Sports, 2=Weather
labels_b = torch.tensor([1, 0, 0, 1, 0])  # 0=Negative, 1=Positive

# Class name mappings
topic_map = {0: "Tech", 1: "Sports", 2: "Weather"}
sentiment_map = {0: "Negative", 1: "Positive"}

# Model & Optimizer
model = MultiTaskSentenceTransformer()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training Loop (1 epoch only for illustration)
model.train()
for epoch in range(1):
    optimizer.zero_grad()

    logits_a, logits_b, _ = model(sample_sentences)

    loss_a = loss_fn(logits_a, labels_a)
    loss_b = loss_fn(logits_b, labels_b)

    total_loss = loss_a + loss_b
    total_loss.backward()
    optimizer.step()

    # Predictions
    preds_a = torch.argmax(logits_a, dim=1)
    preds_b = torch.argmax(logits_b, dim=1)

    acc_a = (preds_a == labels_a).float().mean().item()
    acc_b = (preds_b == labels_b).float().mean().item()

    print(f"\nEpoch {epoch+1}")
    print(f"Loss = {total_loss.item():.4f}, Task A Accuracy = {acc_a:.2f}, Task B Accuracy = {acc_b:.2f}\n")

    print("Detailed predictions per sentence:\n")
    for i, sentence in enumerate(sample_sentences):
        pred_topic = topic_map[preds_a[i].item()]
        true_topic = topic_map[labels_a[i].item()]
        pred_sent = sentiment_map[preds_b[i].item()]
        true_sent = sentiment_map[labels_b[i].item()]

        print(f"Sentence: \"{sentence}\"")
        print(f"  Predicted Topic    : {pred_topic} | True Topic    : {true_topic}")
        print(f"  Predicted Sentiment: {pred_sent}  | True Sentiment: {true_sent}\n")






