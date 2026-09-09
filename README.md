# RAG From Scratch
 
A question-answering system that lets you query your own documents using a local or cloud LLM. Drop in any PDF, TXT, or DOCX file and ask questions about it in plain English. The system retrieves the most relevant passages from your documents and grounds the LLM response in that context rather than relying on the model's parametric memory.
 
---
 
## How it works
 
1. Documents are loaded from the `data/sample_docs/` folder
2. Each document is split into chunks of 500 characters with 50-character overlap using recursive character splitting
3. Every chunk is embedded using `all-MiniLM-L6-v2`, a 384-dimensional sentence transformer that runs locally with no API key
4. Embeddings are stored in a FAISS `IndexFlatL2`; vectors are L2-normalized so dot product distance is equivalent to cosine similarity: `cos(a,b) = 1 - ||a-b||²/2` when `||a||=||b||=1`
5. At query time, the top-3 most similar chunks are retrieved and injected into a prompt template before being sent to the LLM
6. The FAISS index is persisted to disk after the first run — subsequent runs skip re-embedding entirely
---
 
## Setup
 
```bash
conda create -n ragenv python=3.11 -y
conda activate ragenv
pip install -r requirements.txt
```
 
For OpenAI models, add your key to `.env`:
 
```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```
 
For free local inference, install Ollama:
 
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```
 
---
 
## Usage
 
```bash
# interactive Q&A loop
python main.py --model ollama/llama3
 
# single question and exit
python main.py --model ollama/llama3 --question "What is the attention mechanism?"
 
# use OpenAI instead
python main.py --model gpt-3.5-turbo --question "Summarize the document"
 
# retrieve more context chunks (default is 3)
python main.py --model ollama/llama3 --k 5 --question "Explain the encoder-decoder architecture"
 
# debug mode — prints the full prompt sent to the LLM
python main.py --model ollama/llama3 --debug --question "What is multi-head attention?"
 
# point to a different folder of documents
python main.py --model ollama/llama3 --data-dir /path/to/docs
```
 
Add your documents to `data/sample_docs/` before running. Supported formats: PDF, TXT, DOCX.
 
---
 
## Example
 
```bash
wget -O data/sample_docs/attention.pdf https://arxiv.org/pdf/1706.03762
python main.py --model ollama/llama3 --question "What is the attention mechanism?"
```
 
Output:
 
```
Answer:
The attention mechanism relates different positions of a single sequence to compute
a representation of the sequence. It is also referred to as self-attention or intra-attention.
 
Sources used:
  data/sample_docs/attention.pdf, page 12
  data/sample_docs/attention.pdf, page 14
  data/sample_docs/attention.pdf, page 1
```
 
---
 

 
## Project structure
 
```
01-rag-from-scratch/
├── main.py               -- entry point, argument parsing, pipeline orchestration
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/      -- put your documents here
├── faiss_index/          -- auto-created on first run
│   ├── index.faiss
│   └── index.pkl
└── src/
    ├── loader.py         -- document loading (PDF, TXT, DOCX)
    ├── chunker.py        -- recursive character text splitting
    ├── embedder.py       -- sentence transformer embeddings
    ├── vector_store.py   -- FAISS index build, save, load
    ├── retriever.py      -- top-k similarity retrieval
    └── generator.py      -- LLM chain, prompt template, answer generation
```
