# Transformer Modeling and Explainability Analysis

This repository contains the complete implementation for a Transformer-based NLP model fine-tuned for sentiment analysis on the **Amazon Polarity Dataset**. The project emphasizes model interpretability through Attention mechanisms, SHAP, and LIME.

## 📌 1. Objective
The goal of this project is to:
- Fine-tune a **DistilBERT** model for binary sentiment classification.
- Visualize and analyze **Self-Attention** maps to understand token relations.
- Apply **SHAP** and **LIME** to explain individual model predictions.
- Compare the faithfulness, stability, and runtime of different explainability frameworks.

## 🛠️ 2. Requirements & Setup
This project was developed on **Google Colab** using a **Tesla T4 GPU**.

### Libraries used:
To install the necessary dependencies, run:
```bash
pip install transformers datasets shap lime bertviz torch matplotlib pandas scikit-learn

📊 3. Dataset
Source: HuggingFace Amazon Polarity Dataset

Size: A subset of 10,000 samples was used (8,000 for training, 2,000 for testing).

Task: Classifying product reviews as either Positive or Negative.

🚀 4. How to Reproduce Results
Open the NLP_Assignment_04.ipynb notebook in Google Colab.

Ensure the Hardware Accelerator is set to GPU (T4).

Run all cells in sequence (Runtime > Run all).

Attention Visualization: The notebook uses bertviz for an interactive look at Layer 0 and Layer 5 attention heads.

Explainability: 20 random samples are processed through both SHAP and LIME loops to generate comparative explanations.

🧪 5. Methodology & Workflow
Data Pipeline: Tokenization using DistilBERT WordPiece tokenizer.

Model Training: Fine-tuning distilbert-base-uncased with the Trainer API.

Evaluation: Calculation of Accuracy, Precision, Recall, and F1-score.

Attention Analysis: Extraction of weights from multiple layers and heads.

Explanation: Generating local and global importance scores using SHAP and LIME.

Error Analysis: Inspection of misclassified samples (e.g., sarcasm or negations).

📊 6. Results Summary
Model Performance: Achieved an Accuracy and F1-score of ~91%.

Explainability Insight: SHAP demonstrated higher stability and consistency across runs, while LIME proved to be significantly faster for real-time local approximations.

📂 7. Project Structure
NLP_Assignment_04.ipynb: Main Jupyter Notebook with code and outputs.

Transformer_Interpretability_Report.pdf: Detailed 8-12 page research report.

README.md: This documentation file.
