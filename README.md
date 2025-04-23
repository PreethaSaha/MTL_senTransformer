# Multi-Task Sentence Transformer

This repository implements a simple Multi-Task Learning (MTL) model using a Sentence Transformer backbone (BERT). It demonstrates how to:

- Encode sentences into fixed-length embeddings
- Classify sentences for two different tasks (Task A and Task B)
- Train a model using a shared transformer and two task-specific heads

## Tasks
- **Task A**: Sentence classification (e.g., topic or intent)
- **Task B**: Sentiment analysis (e.g., positive, neutral, negative)

### task1.ipynb
- Backbone Model: ```distilbert-base-uncased``` is chosen for its efficiency and performance tradeoff. 
- Pooling Strategy: Mean pooling of token embeddings is used to obtain fixed-length sentence embeddings. It averages only over non-padded tokens using the attention mask.
- Embedding Normalization: Final embeddings are L2 normalized to make them suitable for similarity tasks.

### task2.ipynb 
- Changes made: Multi-task architecture with shared transformer encoder and two task-specific heads
- Task-Specific Heads: Added 2 nn.Linear layers for Task A and Task B
  - Task A Head: A feedforward classification layer for sentence class (e.g., topic)
  - Task B Head: Another classification head for sentiment (positive, negative)
- Shared Encoder: Using the same pretrained transformer ```distilbert-base-uncased``` to produce sentence embeddings for both tasks
- Reused mean-pooling + normalization for embeddings
- Multi-task Loss Handling: You can weight the losses for each task and combine them during training (not shown here since we're only outlining structure)

### task3.txt
 #### All transformer layers and task heads are frozen (no training)
 - Implications: Fast inference only, No learning or adaptation, Useful for static embeddings only
 - Advantages: Evaluation-only or production inference
#### Pretrained transformer is frozen; only task-specific heads are trained
- Implications: Preserves general semantic knowledge, Efficient training, Prevents catastrophic forgetting
- Advantages: Works well for tasks where general language understanding suffices but labels are task-specific
#### Train shared encoder and one head; freeze the other
- Implications: Maintains performance on the frozen task, Helps avoid performance drop during transfer
- Advantages: When reusing the model for a new task without harming performance of an old task

#### Transfer Learning Scenario
Assume applying this model to a new domain-specific multi-task problem (e.g., scientific text classification and sentiment analysis in research abstracts)

- Choice of pre-trained model: ```distilbert-base-uncased``` was used here for speed
  - For domain adaptation, ```allenai/scibert_scivocab_uncased``` is recommended for scientific text
- Freezing strategy:
  - Freeze lower transformer layers to retain general linguistic knowledge
  - Unfreeze upper layers and heads: Adapts to task-specific signals without losing core representations.

#### Rationale
- Lower layers capture general grammar and syntax
- Higher layers adapt to task/domain semantics
- Training heads ensures output space aligns with new labels



### senTrans.py
- Defines the MTL architecture
- Contains a dummy dataset that returns sentences and random labels
- Trains the model, prints predictions and metrics

## Setup
```bash
pip install -r requirements.txt
```

## Running the scripts
```bash
python senTrans.py
```

## Notes
- The dataset is synthetic for demonstration purposes.
- Extend the `DummyMultiTaskDataset` with real datasets for practical use.
- The model uses BERT-base and two linear heads.

## Output
### task1.ipynb
The script prints
-  sentence embeddings of 768 dimensions
-  cosine similarity between the two sentences showcasing semantic encoding

### task2.ipynb
We ran inference on 5 sample sentences. The script prints for each sentence:
- predicted label values for Task A and Task B
- first 5 dimensions of the embeddings

Since no training was performed, predictions are random and not yet meaningful.

### senTrans.py
For each sentence, the model prints:
- Input sentence
- Predicted and true labels and description for both tasks

And for each epoch:
- Loss
- Accuracy for each task
