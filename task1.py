import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', pooling='mean'):
        super(SentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        outputs = self.encoder(**inputs)
        token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        if self.pooling == 'mean':
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sentence_embeddings = sum_embeddings / sum_mask
        elif self.pooling == 'cls':
            sentence_embeddings = token_embeddings[:, 0]
        else:
            raise ValueError("Unsupported pooling method")

        return F.normalize(sentence_embeddings, p=2, dim=1)

if __name__ == '__main__':
    model = SentenceTransformer()
    model.eval()

    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast, dark-colored fox leaps across a sleepy canine.",
        "Transformers are state-of-the-art for NLP tasks."
    ]

    with torch.no_grad():
        embeddings = model(sample_sentences)

    for i, emb in enumerate(embeddings):
        print(f"Sentence {i + 1}:")
        print(emb.numpy()[:5], '...')  # Show first 5 values for brevity

    print("\nCosine Similarities:")
    sim_01 = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
    sim_02 = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0)).item()
    sim_12 = F.cosine_similarity(embeddings[1].unsqueeze(0), embeddings[2].unsqueeze(0)).item()

    print(f"Sentence 1 & 2: {sim_01:.4f}")
    print(f"Sentence 1 & 3: {sim_02:.4f}")
    print(f"Sentence 2 & 3: {sim_12:.4f}")





