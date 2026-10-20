# Evidence-Verified Abstention for Cost-Sensitive Retrieval-Augmented Biomedical QA

Code for the paper **"Evidence-Verified Abstention for Cost-Sensitive Retrieval-Augmented Biomedical Question Answering"**
by Vishal Mehta and Simranpreet Kaur (University Institute of Computing, Chandigarh University).

A training-free gate: an NLI model checks whether the generated rationale is entailed by the retrieved
passages, and a threshold tuned for an error cost λ decides whether to answer or abstain.
Evaluated on PubMedQA with a controlled unanswerable-question protocol (gold abstract removed from the corpus).

## Models and data
| Component | Identifier |
|---|---|
| Retriever | `BAAI/bge-small-en-v1.5` + FAISS (top-3) |
| Generator | `Qwen/Qwen2.5-3B-Instruct` (greedy decoding) |
| Verifier (online) | `cross-encoder/nli-deberta-v3-base` |
| Judge (evaluation only) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Dataset | PubMedQA `pqa_labeled` (1,000 questions), 100 dev / 200 test, seed 42 |

## How to run
1. Open Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Paste `medrag_pipeline.py` cell by cell (each `# CELL n` block is one cell) and run in order.
   Generation (Cell 6) takes ~45–90 min and is checkpointed, so it can be resumed after a disconnect.
3. Outputs are written to `MyDrive/medrag/`: `generations.json`, `scored.csv`, `main_table*.csv`,
   `tradeoff.pdf`, `verifier_separation.pdf`, `manual_check.csv`.

## Key results (test split, N = 200)
At error cost λ = 0.5 the gate reaches selective accuracy 0.68 (13/19 answered answerable questions)
versus 0.32 for standard RAG, with 19% coverage of answerable questions. See the paper for limitations.

## Notes
- This is a research prototype. It is not a medical device and must not be used for clinical decisions.
- Check the licence of each model before reuse (e.g. Qwen2.5-3B has its own licence terms).

## Citation
Add the BibTeX entry here once the paper is published.
