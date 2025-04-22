# Multi-Task Sentence Transformer

This repository implements a simple Multi-Task Learning (MTL) model using a Sentence Transformer backbone (BERT). It demonstrates how to:

- Encode sentences into fixed-length embeddings
- Classify sentences for two different tasks (Task A and Task B)
- Train a model using a shared transformer and two task-specific heads

## Tasks
- **Task A**: Sentence classification (e.g., topic or intent)
- **Task B**: Sentiment analysis (e.g., positive, neutral, negative)

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
For each sentence, the model prints:
- Input sentence
- Predicted and true labels and description for both tasks

And for each epoch:
- Loss
- Accuracy for each task
