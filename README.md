# Financial Sentiment Classification using FinBERT

This repository contains an end-to-end NLP classification project for financial sentiment analysis. The objective is to classify financial text into three sentiment categories: **negative**, **neutral**, and **positive**. The project compares a traditional machine learning baseline model against a domain-specific transformer model, FinBERT.

## Project Summary

Financial text is often difficult to classify because the sentiment is highly context-dependent. A word that appears negative in general language may have a positive meaning in finance. For example, a phrase such as "costs fell" can indicate improved profitability even though the word "costs" may appear negative on its own.

To address this, the project fine-tunes **FinBERT**, a BERT-based transformer model pretrained on financial text. Its performance is compared against a baseline model using **TF-IDF** features and **Logistic Regression**.

## Dataset

The dataset used is **Financial PhraseBank**, loaded from Hugging Face Datasets. The selected subset is:

```python
financial_phrasebank / sentences_allagree
```

The `sentences_allagree` subset was chosen because it contains examples where all human annotators agreed on the sentiment label, making it a higher-quality subset for supervised learning.

The dataset is loaded using:

```python
dataset = load_dataset(
    "financial_phrasebank",
    "sentences_allagree",
    trust_remote_code=True
)
```

The labels are:

| Label | Sentiment |
|---:|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

## Repository Files

```text
.
├── FinBERT_classification.ipynb   # Main notebook containing the full implementation
├── answer.docx                    # Project report/write-up
└── README.md                      # Project documentation
```

## Methodology

The notebook follows a complete machine learning workflow. First, the dataset is loaded and converted into a pandas DataFrame. The relevant columns are standardized to `text` and `label`. The dataset is then split into training, validation, and test sets using an 80/10/10 split with stratified sampling to preserve the class distribution.

A baseline model is trained using TF-IDF vectorization with a maximum of 5000 features, followed by a Logistic Regression classifier. This provides a simple and interpretable benchmark.

The main model uses `ProsusAI/finbert`. The text is tokenized using the FinBERT tokenizer with padding and truncation applied to a maximum sequence length of 128 tokens. The FinBERT model is then fine-tuned for three-class sequence classification.

## Training Configuration

| Parameter | Value |
|---|---:|
| Model | `ProsusAI/finbert` |
| Learning Rate | `2e-5` |
| Batch Size | `16` |
| Epochs | `5` |
| Weight Decay | `0.01` |
| Early Stopping | Patience = `2` |
| Evaluation Strategy | Per epoch |
| Max Sequence Length | `128` |

## Evaluation Metrics

The models are evaluated using accuracy, precision, recall, and F1-score. Macro-averaged F1-score is especially important because it treats all classes equally, regardless of class imbalance.

## Results

### Baseline: TF-IDF + Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 0.8502 |
| Macro F1-score | 0.79 |
| Weighted F1-score | 0.84 |

The baseline model performs reasonably well, especially on the neutral class. However, it struggles more with the negative and positive classes because TF-IDF mainly relies on word frequency and does not capture deeper contextual meaning.

### FinBERT

| Metric | Score |
|---|---:|
| Accuracy | 0.9692 |
| Macro F1-score | 0.94 |
| Weighted F1-score | 0.97 |

FinBERT significantly outperforms the baseline model. The improvement shows the benefit of using a domain-specific transformer model for financial NLP, as FinBERT is better able to capture semantic relationships and subtle financial cues.

## Class-wise FinBERT Performance

| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.85 | 0.94 | 0.89 | 31 |
| 1 | 1.00 | 1.00 | 1.00 | 139 |
| 2 | 0.96 | 0.91 | 0.94 | 57 |

The model performs best on the neutral class, achieving perfect precision, recall, and F1-score. The negative and positive classes also achieve strong performance, although a small number of errors remain between these two sentiment categories.

## Confusion Matrix Summary

| Actual Class | Correct Predictions | Main Errors |
|---|---:|---|
| Negative | 29 / 31 | 2 predicted as positive |
| Neutral | 139 / 139 | No errors |
| Positive | 52 / 57 | 5 predicted as negative |

The confusion matrix shows that no neutral examples were misclassified. Most of the remaining errors occurred between positive and negative examples, which suggests that some financial sentences contain mixed or ambiguous signals.

## Error Analysis

Most misclassifications are caused by the context-dependent nature of financial language. For example:

```text
Unit costs for flight operations fell by 6.4 percent.
```

This sentence was labelled as positive but predicted as negative. The model may have associated the word "costs" with negative sentiment, even though the sentence actually describes a reduction in costs, which is generally a positive financial signal.

This highlights one of the main challenges in financial NLP: sentiment is often implied through business context rather than expressed through obviously positive or negative words.

## Key Findings

FinBERT outperformed the TF-IDF baseline by a clear margin, improving accuracy from approximately 0.85 to 0.97. This demonstrates the importance of domain-specific pretraining for financial language understanding. The results also show that neutral financial statements are easier to classify in this dataset, while positive and negative examples can be more ambiguous due to mixed financial signals.

## Limitations

Although FinBERT achieved strong results, there are still limitations. The dataset is relatively small for deep learning, and the `sentences_allagree` subset may be easier than noisier real-world financial text because all annotators agreed on the labels. Financial sentiment can also depend on temporal and macroeconomic context, which is not included in this text-only classification setup.

## Future Work

Future improvements could include training on a larger and more diverse financial text dataset, comparing FinBERT with other transformer models such as BERT, RoBERTa, or DeBERTa, and incorporating numerical financial indicators such as stock prices, interest rates, or macroeconomic variables. The project could also be extended into a real-time system for monitoring financial news sentiment.

## Installation

Install the required dependencies with:

```bash
pip install transformers datasets scikit-learn pandas torch matplotlib seaborn
```

Depending on your environment, you may also need:

```bash
pip install accelerate
```

## How to Run

Open the notebook:

```text
FinBERT_classification.ipynb
```

Then run the cells from top to bottom. The notebook will load the dataset, train the baseline model, fine-tune FinBERT, evaluate both models, plot the confusion matrix, and display sample misclassifications.

## Notes

When loading the Financial PhraseBank dataset, Hugging Face may require `trust_remote_code=True` because the dataset uses custom loading code. This is already included in the notebook.

Some Jupyter progress bars or notebook outputs may display differently depending on whether the notebook execution was interrupted or restarted. The important final outputs are the classification reports, model comparison, confusion matrix, and error analysis.

## Conclusion

This project demonstrates that FinBERT is highly effective for financial sentiment classification. Compared with a TF-IDF and Logistic Regression baseline, FinBERT achieves stronger performance by capturing the contextual and domain-specific meaning of financial language.
