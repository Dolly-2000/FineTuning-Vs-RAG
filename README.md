# Fine-Tuning vs. RAG: A Comparative Analysis on SQuAD

This project provides a hands-on, comparative analysis between two popular techniques for adapting Large Language Models (LLMs) to specific tasks: **QLoRA Fine-Tuning** and **Retrieval-Augmented Generation (RAG)**.

The goal is to evaluate their performance on the Stanford Question Answering Dataset (SQuAD) using the `mistralai/Mistral-7B-Instruct-v0.2` model as the backbone.

## Project Summary & Key Insight

This analysis reveals a critical distinction between the two methods:

-   **▶ Fine-Tuning teaches a model a new SKILL or BEHAVIOR.**
    Our model, fine-tuned with QLoRA, became a **specialist** at the SQuAD task. It learned to generate the precise, extractive, short-form answers that the SQuAD metrics reward, resulting in a significantly higher F1 and Exact Match score.

-   **▶ RAG provides a model with new KNOWLEDGE.**
    The zero-shot RAG system had access to the correct facts, but its performance was penalized because it provided more natural, conversational, or explanatory answers, which do not strictly match the ground-truth format.

| System | Primary Use Case | Knowledge Handling | Update Cost | Format Specialization |
| :--- | :--- | :--- | :--- | :--- |
| **Fine-Tuning (QLoRA)** | **Skill / Behavior** | Static / Becomes Stale | Expensive (Re-train) | **Extremely High** |
| **RAG** | **Knowledge Provisioning** | **Dynamic / Always Fresh** | Cheap (Re-index) | Low |


## Final Quantitative Results

Evaluation was performed on 200 examples from the SQuAD validation set. The fine-tuned model demonstrates superior performance in mastering the specific question-answering format.

| System | Exact Match (%) | F1 Score (%) |
| :--- | :---: | :---: |
| RAG (Zero-Shot) | 81.50 | 82.64 |
| **Fine-Tuning (QLoRA @ 500 steps)** | **94.00** | **95.67** |


## Tech Stack & Models

- **Core LLM:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Embedding Model (RAG):** `all-MiniLM-L6-v2`
- **Fine-Tuning:** `peft` (QLoRA), `bitsandbytes` (4-bit quantization), `trl` (SFTTrainer)
- **Vector Store (RAG):** `faiss-cpu`
- **Core Libraries:** PyTorch, Transformers, Datasets, Evaluate
