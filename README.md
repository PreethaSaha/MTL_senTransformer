# Multi-Task Learning Sentence Transformer

This repository implements a simple Multi-Task Learning (MTL) model using a Sentence Transformer backbone (BERT). It demonstrates how to:

- Encode sentences into fixed-length embeddings
- Classify sentences for two different tasks (Task A and Task B)
- Train a model using a shared transformer and two task-specific heads

## Tasks
- **Task A**: Sentence classification (e.g., topic or intent)
- **Task B**: Sentiment analysis (e.g., positive, neutral, negative)

### task1.py
- Backbone Model: ```distilbert-base-uncased``` is chosen for its efficiency and performance tradeoff. 
- Pooling Strategy: Mean pooling of token embeddings is used to obtain fixed-length sentence embeddings. It averages only over non-padded tokens using the attention mask.
- Embedding Normalization: Final embeddings are L2 normalized to make them suitable for similarity tasks.

### task2.py 
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



### task4.py
- Data: simulated batch of tokenized text + labels for Task A and Task B
- Forward Pass: inputs pass through shared encoder, then to shared pooling and finally to two task heads
- Loss Function: separate ```CrossEntropyLoss``` per task; combined loss = weighted sum
- Metrics: Per-task accuracy (classification accuracy on logits)
- Optimization: Backprop from joint loss; both heads and (optionally) encoder updated
- Assumption: All tasks are defined over the same input text

Please refer to task4.txt for the summary of training loop implementation

## Setup

To reproduce the outputs, follow these steps:

1. Clone the repository

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>git clone https://github.com/PreethaSaha/MTL_senTransformer.git
  </code></pre>
</div>

2. Navigate to the project directory

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>cd MTL_senTransformer
  </code></pre>
</div>

3. Create and activate an environment

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>python3 -m venv venv        # Create a virtual environment named 'venv'
source venv/bin/activate    # Activate the virtual environment
</code></pre>
</div>

Please ensure all the notebook files are in the same directory.

4. Install the required dependencies

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code> pip install -r requirements.txt
  </code></pre>
</div>


## Output
### task1.py
The script prints
-  sentence embeddings of 768 dimensions
-  cosine similarity between the two sentences showcasing semantic encoding

### task2.py
We ran inference on 5 sample sentences. The script prints for each sentence:
- predicted label values for Task A and Task B
- first 5 dimensions of the embeddings

Since no training was performed, predictions are random and not yet meaningful.

### task4.py
For each sentence, the script prints:
- Input sentence
- Predicted and true labels and description for both tasks

And for each epoch:
- Loss
- Accuracy for each task

Low accuracy is expected with 5 sentences and 1 epoch. This setup is useful to test code structure, not model performance. To get meaningful accuracy, one has to use a real dataset, train for multiple epochs with proper data splits, and evaluate on validation/test data for realistic metrics.
