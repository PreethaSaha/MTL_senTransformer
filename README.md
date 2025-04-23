# Multi-Task Sentence Transformer

This repository implements a simple Multi-Task Learning (MTL) model using a Sentence Transformer backbone (BERT). It demonstrates how to:

- Encode sentences into fixed-length embeddings
- Classify sentences for two different tasks (Task A and Task B)
- Train a model using a shared transformer and two task-specific heads

## Tasks
- **Task A**: Sentence classification (e.g., topic or intent)
- **Task B**: Sentiment analysis (e.g., positive, neutral, negative)

## task1.ipynb
- Backbone Model: ```distilbert-base-uncased``` is chosen for its efficiency and performance tradeoff. 
- Pooling Strategy: Mean pooling of token embeddings is used to obtain fixed-length sentence embeddings. It averages only over non-padded tokens using the attention mask.
- Embedding Normalization: Final embeddings are L2 normalized to make them suitable for similarity tasks.

## senTrans.py
- Defines the MTL architecture
- Contains a dummy dataset that returns sentences and random labels
- Trains the model, prints predictions and metrics

## Setup
```bash
pip install -r requirements.txt
```

## Running the Training Script
```bash
python senTrans.py
```

## Notes
- The dataset is synthetic for demonstration purposes.
- Extend the `DummyMultiTaskDataset` with real datasets for practical use.
- The model uses BERT-base and two linear heads.

## Output
### task1.ipynb
-  sentence embeddings of 768 dimensions
-  cosine similarity between the two sentences showcasing semantic encoding

### senTransformer.py
For each sentence, the model prints:
- Input sentence
- Predicted and true labels and description for both tasks

And for each epoch:
- Loss
- Accuracy for each task
