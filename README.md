# localized-llm-market-access

Markdown
# Localized LLM Payer Coverage Navigation
**Evaluating Localized Large Language Models and Retrieval-Augmented Generation for Payer Coverage Navigation in US Market Access**

This repository contains the complete empirical framework, dataset, and evaluation scripts.

## 📌 Repository Overview
This project evaluates the operational speed and clinical accuracy of deploying localized Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to interpret complex health insurance prior authorization (PA) policies. 

By avoiding public cloud APIs (e.g., ChatGPT), this architecture complies with HIPAA and enterprise data security constraints. The study establishes scaling laws across 1.5-Billion, 7-Billion, and 14-Billion parameter models (Qwen2.5 family).

## 📂 Directory Structure
```text
├── data/
│   ├── golden_dataset/     # N=60 Clinical Scenario Matrix & Graded Results
│   └── payer_policies/     # Original Commercial PA Policy PDFs
├── scripts/
│   ├── 01_model_1b_run.py  # Local 1.5B RAG pipeline
│   ├── 02_model_7b_run.py  # Local 7B 4-bit Quantized RAG pipeline
│   ├── 03_model_14b_run.py # Local 14B VRAM Hardware Stress Test
│   └── 04_auto_grader.py   # Automated NLP String-Matching Evaluator
├── requirements.txt        # Exact software dependencies for reproducibility
└── README.md
⚙️ Hardware Requirements
Standard Pipelines (Scripts 01 & 02): Minimum 15GB VRAM (e.g., NVIDIA T4, A10G). Tested successfully on Google Colab (Free Tier).

Stress Test (Script 03): Deliberately triggers an Out-of-Memory (OOM) error on standard 15GB infrastructure to demonstrate the hardware barrier of scaling to 14B parameters.

🚀 Step-by-Step Replication Guide
To ensure exact replication of the latency metrics, zero-hallucination barriers, and hardware faults described in the manuscript, follow these steps:

1. Environment Setup
Clone the repository and install the frozen dependencies to prevent version drift:

Bash
git clone [https://github.com/](https://github.com/)[Your-Username]/localized-llm-market-access.git
cd localized-llm-market-access
pip install -r requirements.txt
2. Run the Empirical Baselines
Execute the Python scripts in order. Note: All scripts utilize a fixed random seed (seed=42) to control stochasticity in the vector database and token generation.

Bash
# Run the lightweight 1.5B model test
python scripts/01_model_1b_run.py

# Run the highly compliant 7B model test (Requires 4-bit quantization)
python scripts/02_model_7b_run.py
3. Run the Evaluation and Statistical Grader
Once the raw AI responses are generated, run the NLP evaluation script. This will extract the definitive binary clinical outcomes, compare them to the Ground Truth matrix, and run the paired t-tests for latency.

Bash
python scripts/04_auto_grader.py
Output files will be saved in /data/golden_dataset/.

📜 Code Availability Statement
All code and datasets are provided as open-source under the MIT License to facilitate peer review and independent replication of the findings.

For peer reviewers: If you encounter environment conflicts due to CUDA driver mismatch, please ensure your PyTorch build matches the hardware accelerators available on your testing node.
"""

with open("requirements.txt", "w") as f:
f.write(req_content)
