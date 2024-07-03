# Impact of Attention Mechanism on Question Answering in Transformers

## Project Overview
This research explores the understanding of attention mechanisms in Transformer-based models, specifically BERT, to improve their performance on the question-answering task. We aim to identify the optimal number of attention heads that balance accuracy and computational efficiency. By contrasting the fine-tuned BERT model with varying attention heads against an LSTM model without attention mechanisms, we provide insights into the effective deployment of attention mechanisms in language models.

## Abstract
Our research question delves into how BERT’s attention mechanism enhances its ability to accurately interpret and respond to questions across a diverse array of subjects compared to LSTM’s sequential data processing, particularly in tasks that require a deep understanding of context and nuanced language interpretation. We assess the impact of attention on handling complex queries when context is provided, specifically in the context of extractive question answering.

## Methodology

### Dataset Overview
We utilized the SQuAD (Stanford Question Answering Dataset) v2 for our experiments. This dataset includes questions that are purposely unanswerable based on the given context, challenging the model to identify when no valid answer exists.

### Data Preprocessing

#### LSTM
- Tokenization of textual components.
- Padding to ensure uniform sequence lengths.
- Conversion of tokenized words into dense vector representations using word embeddings.
- Establishing input-output pairs for LSTM training.

#### BERT
- Tokenization using the BERT tokenizer.
- Padding and truncation to standardize sequence length.
- Generation of attention masks and segment IDs.
- Labeling of answer spans within the tokenized context.

### Model Implementation and Fine Tuning

#### LSTM Experiment
Implemented a Bi-LSTM model augmented with dropout layers to mitigate overfitting. Trained the model over 20 epochs and evaluated its performance using F1 and Exact Match scores.

#### BERT Experiment
Utilized a pre-trained BERT-base model with default 12 attention heads. Trained and validated the model over 3 epochs, recording loss and evaluating the model using F1 and Exact Match scores.

#### Attention Heads Experiment
Conducted experiments using Optuna to determine the optimal number of attention heads. Evaluated configurations ranging from 4 to 16 attention heads to find the best balance between performance and computational efficiency.

#### Unanswerable Questions Experiment
Separated unanswerable questions from the dataset and evaluated the model's confidence in predicting "no answer" using a strict threshold.

## Results

### Evaluation Metrics
- **Exact Match**: Proportion of predictions that exactly match any one of the ground truth answers.
- **F1 Score**: Harmonic mean of precision and recall, providing a comprehensive assessment of the model’s ability to accurately identify answer spans within passages.

### Analysis
- **LSTM**: Achieved an accuracy of 47% (F1) and 34% (Exact Match) over 20 epochs.
- **BERT**: Achieved an accuracy of 73% (F1) and 59% (Exact Match) over 3 epochs with 12 attention heads.
- **Attention Heads**: Optimal number of attention heads found to be 12, balancing model complexity and performance.

## Conclusion and Future Scope
Our findings indicate that BERT models with attention mechanisms significantly outperform LSTM models on question-answering tasks. However, increasing the number of attention heads beyond a certain threshold leads to diminishing returns due to overfitting. Future work will explore alternative attention mechanisms to further enhance model performance while reducing computational complexity.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](http://arxiv.org/abs/1810.04805)
- [Attention is All You Need](http://arxiv.org/abs/1806.03822)
- [SQuAD v2 Dataset](https://huggingface.co/datasets/rajpurkar/squad_v2)

## Team Members
- Manoj Virinchi Chitta
- Tarun Reddy Nimmala
- Sai Aakash Ekbote
- Satya Aakash
- Rishika Reddy

## Project Links
- [GitHub Repository](https://github.com/saekbote/cai6307-parsingpandas)
- [SQuAD v2 Dataset](https://huggingface.co/datasets/rajpurkar/squad_v2)
