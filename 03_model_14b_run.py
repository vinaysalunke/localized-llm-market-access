import os
import time
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import random
from transformers import BitsAndBytesConfig
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def set_reproducibility_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_reproducibility_seed(42)

print("Step 1: Reading and Chunking Payer Policies...")
loader = PyPDFDirectoryLoader("data/payer_policies/")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(raw_documents)

print("Step 2: Building the Vector Database...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = FAISS.from_documents(chunked_docs, embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

print("Step 3: Loading Qwen 14B Model Locally...")
model_id = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True), device_map="auto")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=150, 
    temperature=0.1,
    do_sample=True,
    return_full_text=False
)

def parse_dataset_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    scenarios = []
    blocks = raw_text.split("[Scenario ")
    for block in blocks[1:]:
        lines = block.split("\n")
        metadata = lines[0].split("]")[0].strip()
        q_idx = block.find("Question: ")
        gt_idx = block.find("Ground_Truth: ")
        question = block[q_idx+10:gt_idx].strip()
        ground_truth = block[gt_idx+13:].strip()
        scenarios.append({"Metadata": metadata, "Question": question, "Ground_Truth": ground_truth})
    return scenarios

print("Step 4: Loading Dataset...")
test_cases = parse_dataset_file("data/golden_dataset/Psoriasis_Biologics_60_Question_Matrix.txt")

results_data = []
print("Step 5: Running experiments...")

for case in tqdm(test_cases):
    q = case["Question"]
    gt = case["Ground_Truth"]
    
    baseline_messages = [
        {"role": "system", "content": "You are a health insurance prior authorization reviewer. Answer the user's question accurately based on general knowledge."},
        {"role": "user", "content": q}
    ]
    b_prompt = tokenizer.apply_chat_template(baseline_messages, tokenize=False, add_generation_prompt=True)
    
    start = time.time()
    b_out = pipe(b_prompt)[0]['generated_text']
    b_lat = round(time.time() - start, 2)
    
    retrieved_docs = retriever.invoke(q)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    rag_messages = [
        {"role": "system", "content": "You are a health insurance prior authorization reviewer. Answer the query accurately using ONLY the facts in the provided Context. If the context does not explicitly contain the answer, state 'The policy does not specify.'"},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{q}"}
    ]
    r_prompt = tokenizer.apply_chat_template(rag_messages, tokenize=False, add_generation_prompt=True)

    start = time.time()
    r_out = pipe(r_prompt)[0]['generated_text']
    r_lat = round(time.time() - start, 2)
    
    results_data.append({
        "Category": case["Metadata"],
        "Question": q,
        "Ground_Truth_Policy": gt,
        "Baseline_AI_Response": b_out.strip(),
        "Baseline_Latency_Sec": b_lat,
        "Optimized_RAG_Response": r_out.strip(),
        "RAG_Latency_Sec": r_lat
    })

df = pd.DataFrame(results_data)
df.to_excel("data/golden_dataset/Maximum_14B_RAG_Results.xlsx", index=False)
print("Saved to Maximum_14B_RAG_Results.xlsx")
