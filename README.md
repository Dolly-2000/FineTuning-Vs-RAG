Project: Fine-Tuning vs. Retrieval-Augmented Generation (RAG)
This repository contains a Jupyter Notebook that conducts a detailed comparison between two popular techniques for enhancing Large Language Models (LLMs): QLoRA Fine-Tuning and Retrieval-Augmented Generation (RAG). The experiment uses the Stanford Question Answering Dataset (SQuAD) to evaluate which method is more effective for extractive question-answering tasks.
Project Goal
The primary objective is to quantitatively compare the performance of a powerful base model (mistralai/Mistral-7B-Instruct-v0.2) when adapted for a specific task using two different methods:
RAG: Providing the model with relevant context "on-the-fly" from a pre-indexed knowledge base (zero-shot).
Fine-Tuning: Adjusting the model's weights on a task-specific dataset using Parameter-Efficient Fine-Tuning (PEFT) with QLoRA.
The project measures performance using standard question-answering metrics: Exact Match (EM) and F1 Score.
Key Findings & Results
The experiment demonstrates that for the SQuAD dataset, the fine-tuned model significantly outperforms the RAG system.
Quantitative Comparison
System	Exact Match (%)	F1 Score (%)
RAG (Zero-Shot)	81.50	82.64
Fine-Tuning (QLoRA @ 500 steps)	94.00	95.67
Analysis
The results highlight a fundamental difference between the two approaches:
Fine-Tuning teaches a model a new SKILL or BEHAVIOR. By training on 5,000 SQuAD examples, the model became a specialist at the task, learning to generate the precise, extractive answers that the benchmark rewards.
RAG provides a model with new KNOWLEDGE. The RAG model had access to the correct facts via its retrieval mechanism but was penalized for generating more natural, conversational responses, which scored lower on the strict extractive metrics of SQuAD.
Methodology
1. RAG System
The RAG pipeline was constructed as follows:
Knowledge Base: The contexts from the SQuAD training and validation sets were used as the knowledge source.
Text Splitting: Contexts were split into chunks of 512 characters with an overlap of 64 characters.
Embedding Model: The all-MiniLM-L6-v2 sentence transformer was used to generate embeddings for all text chunks.
Vector Store: A FAISS (IndexFlatL2) index was built to store the embeddings for efficient similarity search.
Generator: The mistralai/Mistral-7B-Instruct-v0.2 model, loaded in 4-bit precision using bitsandbytes, served as the language model to generate answers based on the retrieved context.
2. Supervised Fine-Tuning (SFT) with QLoRA
The fine-tuning process was configured as follows:
Base Model: The same 4-bit quantized mistralai/Mistral-7B-Instruct-v0.2 was used as the starting point.
Dataset Formatting: 5,000 examples from the SQuAD training set were formatted into a specific instruction-following prompt structure.
QLoRA Configuration:
Rank (r): 16
Alpha (lora_alpha): 32
Target Modules: q_proj and v_proj attention layers.
Training: The model was trained using the SFTTrainer from the TRL library for one epoch. Due to hardware constraints in the Colab environment, the full training run did not complete. The final evaluation was performed on the model adapter saved at checkpoint-500.
