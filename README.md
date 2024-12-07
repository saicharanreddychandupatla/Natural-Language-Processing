# Natural-Language-Processing
# Plagiarism Detection Using Transformers

## Authors
- *Vaishnavi Kukkala* (vkukk2@unh.newhaven.edu)  
- *Hari Krishna Para* (hpara2@unh.newhaven.edu)  
- *Sai Charan Chandu Patla* (schan30@unh.newhaven.edu)  
*University of New Haven, Department of Data Science*

---

## Overview

Plagiarism detection is a significant challenge in academic and professional contexts. This project leverages cutting-edge transformer-based models to accurately detect plagiarized content, including verbatim copying, paraphrasing, and semantic rewording. Key models evaluated include *BERT, **RoBERTa, **T5, and a **BERT+LSTM hybrid*, trained and fine-tuned on plagiarism-specific datasets.

### Key Results:
- *BERT*: 85.87% accuracy on Dataset 1 (D1), 83.77% on Dataset 2 (D2).  
- *BERT+LSTM Hybrid*: 85.20% (D1), 82.84% (D2).  
- *RoBERTa*: 84.46% (D1), 82.03% (D2).  
- *T5*: 77.84% (D1), 66.49% (D2).

---

## Features
- *Transformer-Based Models*: Fine-tuned versions of BERT, RoBERTa, and T5.  
- *Hybrid Architectures*: Combines BERT's contextual embeddings with LSTM for sequential dependencies.  
- *Multi-Dataset Testing*: Evaluated on SNLI (D1) and MRPC (D2).  
- *Preprocessing Pipeline*: Tokenization, padding, truncation, and special token handling for consistency.

---

## Methodology

### Datasets:
1. *SNLI (Stanford Natural Language Inference)*: Modified to detect semantic relationships indicative of plagiarism.  
2. *MRPC (Microsoft Research Paraphrase Corpus)*: Focuses on sentence paraphrasing.  

### Preprocessing Steps:
- *Tokenization*: Model-specific tokenizers (BERT, RoBERTa, T5).  
- *Special Tokens*: Added to represent sequence structure.  
- *Padding and Truncation*: Standardized input lengths for batch processing.  
- *Vectorization*: Embedding vectors generated from tokenized text.

### Models:
- *BERT*: Bidirectional Transformer for semantic understanding.  
- *RoBERTa*: Optimized version of BERT with enhanced training.  
- *BERT+LSTM Hybrid*: Combines BERT embeddings with LSTM's sequential analysis.  
- *T5*: Treats tasks as text-to-text problems for flexibility.  

### Training Configuration:
- *Loss Function*: Binary cross-entropy for classification tasks.  
- *Batch Size*: 16 (optimized for GPU memory).  
- *Epochs*: 5 (determined to avoid overfitting).  
- *Metrics*: Accuracy, precision, recall, and F1-score.  

---
## How to Use?

This project includes a deep learning-based notebook for detecting plagiarism using transformer models like BERT, RoBERTa, T5, and a hybrid BERT+LSTM architecture. Follow the instructions below to use the notebook:

1. *Access the Notebook*:  
   Download or clone the repository and open the notebook file:  
   [Plagarism_Detection_code.ipynb](https://github.com/saicharanreddychandupatla/Plagiarism-Detection-using-Transformers/blob/main/Plagarism_Detection_code.ipynb).

2. *Setup Environment*:  
   Open the notebook in Google Colab or Jupyter Notebook. Ensure that you have Python 3.12 installed if using a local setup.

3. *Dataset Preparation*:  
   - Download the datasets (SNLI and MRPC) provided in the code using the Hugging Face datasets library.
   - Alternatively, add a shortcut of your custom dataset folder to your Google Drive for easy access.

4. *Run the Notebook*:  
   - Follow the sequential execution of cells in the notebook.
   - The notebook covers:
     - Installation of required dependencies (transformers, datasets, pandas).
     - Loading and preprocessing of datasets.
     - Training and evaluation of transformer models for plagiarism detection.

5. *Customize*:  
   - Modify hyperparameters (batch size, learning rate, etc.) and the dataset paths if necessary.
   - Experiment with additional models or datasets for comparative analysis.

6. *Save Results*:  
   - Evaluation metrics such as accuracy, precision, recall, and F1-score are displayed during training. Save any desired outputs or model checkpoints.


---

## Results
- *Best Overall Model*: BERT, achieving top accuracy and F1 scores on both datasets.  
- *Hybrid Model*: Demonstrated robust performance, excelling in detecting sequential patterns.  
- *T5*: Struggled with nuanced paraphrasing on MRPC dataset.  

| Model        | Accuracy (D1) | Accuracy (D2) |
|--------------|---------------|---------------|
| *BERT*     | 85.87%        | 83.77%        |
| *BERT+LSTM*| 85.20%        | 82.84%        |
| *RoBERTa*  | 84.46%        | 82.03%        |
| *T5*       | 77.84%        | 66.49%        |

---

## Conclusion and Future Work

This project demonstrates the potential of transformers in plagiarism detection, with models like BERT achieving state-of-the-art results. Future work includes:
- *Domain-Specific Models*: Fine-tuning on specialized datasets.  
- *Real-Time Detection*: Developing scalable, real-time solutions.  
- *Multilingual Support*: Enhancing model versatility for global applications.

---

## Limitations
- *High Computational Costs*: Transformer models require substantial resources.  
- *Generalization*: Domain-specific and multilingual adaptations remain challenging.

---

## References
1. S. V. Moravvej et al., "A Novel Plagiarism Detection Approach Combining BERT-based Word Embedding..."  
2. R. Patil et al., "A Novel Natural Language Processing Based Model for Plagiarism Detection."  
3. [Additional references available in the project documentation.](https://arxiv.org)
