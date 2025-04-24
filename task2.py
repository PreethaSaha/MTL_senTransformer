import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels_task_a=3, num_labels_task_b=2):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size
        self.pooling = 'mean'

        self.classifier_task_a = nn.Linear(hidden_size, num_labels_task_a)  # Task A: sentence classification
        self.classifier_task_b = nn.Linear(hidden_size, num_labels_task_b)  # Task B: sentiment analysis

    def mean_pooling(self, outputs, attention_mask):
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        outputs = self.encoder(**inputs)
        sentence_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        logits_task_a = self.classifier_task_a(sentence_embeddings)
        logits_task_b = self.classifier_task_b(sentence_embeddings)

        return logits_task_a, logits_task_b, sentence_embeddings

if __name__ == '__main__':
    # Instantiate model
    model = MultiTaskSentenceTransformer()
    model.eval()

    # Sample sentences
    sentences = [
        "It's sunny today.",                                      # Task A: Weather, Task B: Positive
        "This laptop has great battery life.",                    # Task A: Technology, Task B: Positive
        "The football match was disappointing.",                  # Task A: Sports, Task B: Negative
        "AI is revolutionizing healthcare and finance.",          # Task A: Technology, Task B: Positive
        "Rain dampened the final leg of the cycling tournament."  # Task A: Sports, Task B: Negative
    ]

    # Run inference
    with torch.no_grad():
        logits_a, logits_b, embeddings = model(sentences)
        preds_a = torch.argmax(logits_a, dim=1)
        preds_b = torch.argmax(logits_b, dim=1)

    # Task label names
    task_a_labels = {0: 'Technology', 1: 'Weather', 2: 'Sports'}
    task_b_labels = {0: 'Negative', 1: 'Positive'}

    # Display results
    print("\n=== Multi-Task Model Predictions ===")
    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")
        print(f"→ Predicted Topic (Task A): {task_a_labels[preds_a[i].item()]} [{preds_a[i].item()}]")
        print(f"→ Predicted Sentiment (Task B): {task_b_labels[preds_b[i].item()]} [{preds_b[i].item()}]")
        print(f"→ Embedding Preview (first 5 dims): {embeddings[i][:5].numpy()} ...")





