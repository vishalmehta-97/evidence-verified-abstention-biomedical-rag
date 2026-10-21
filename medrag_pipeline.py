# ============================================================
# Evidence-Verified Abstention for Biomedical RAG
# Full pipeline — paste each "# CELL n" block into its own
# Google Colab cell, in order, OR paste the whole file into
# ONE cell and run it (it will execute top to bottom).
#
# Runtime > Change runtime type > T4 GPU  (do this FIRST)
# ============================================================

# ============================================================
# CELL 1 — Install + mount Drive (run once)
# ============================================================
get_ipython().system('pip -q install -U transformers accelerate sentence-transformers faiss-cpu datasets scikit-learn "pandas<2.4" matplotlib tqdm')
from google.colab import drive
drive.mount('/content/drive')
import os
OUT = "/content/drive/MyDrive/medrag"
os.makedirs(OUT, exist_ok=True)
print("Output folder:", OUT)

# ============================================================
# CELL 2 — Imports + config
# ============================================================
import re, json, random, numpy as np, pandas as pd, torch
from datasets import load_dataset
from tqdm.auto import tqdm

SEED, N_DEV, N_TEST, TOP_K = 42, 100, 200, 3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available(), "GPU not enabled! Go to Runtime > Change runtime type > T4 GPU"
print("GPU OK:", torch.cuda.get_device_name(0))

# ============================================================
# CELL 3 — Load PubMedQA + build the unanswerable-question protocol
# ============================================================
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")  # 1000 expert-labelled Qs

ids = list(range(len(ds))); random.shuffle(ids)
dev_ids, test_ids = ids[:N_DEV], ids[N_DEV:N_DEV + N_TEST]
dev_set = set(dev_ids)

def tag(split):
    # alternate answerable / unanswerable within each split
    return {i: ("unanswerable" if k % 2 else "answerable") for k, i in enumerate(split)}

status = {**tag(dev_ids), **tag(test_ids)}
removed = {i for i, s in status.items() if s == "unanswerable"}

corpus, src = [], []  # passage-level corpus; gold abstracts of "unanswerable" Qs are excluded
for i, ex in enumerate(ds):
    if i in removed:
        continue
    for p in ex["context"]["contexts"]:
        corpus.append(p)
        src.append(i)

print("Corpus passages:", len(corpus), "| Abstracts removed:", len(removed))
print(pd.Series([ds[i]["final_decision"] for i in dev_ids + test_ids]).value_counts())

# ============================================================
# CELL 4 — Retriever: BGE embeddings + FAISS index
# ============================================================
import faiss
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
E = emb.encode(corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(E.shape[1])
index.add(E)
QP = "Represent this sentence for searching relevant passages: "

def retrieve(q, k=TOP_K):
    qv = emb.encode([QP + q], normalize_embeddings=True).astype("float32")
    s, ix = index.search(qv, k)
    return [corpus[j] for j in ix[0]], s[0].tolist(), [src[j] for j in ix[0]]

# sanity check: retrieval recall@k on answerable dev questions — report this in the paper
hits = [int(i in retrieve(ds[i]["question"])[2]) for i in dev_ids if status[i] == "answerable"]
print("Recall@%d (answerable dev): %.2f" % (TOP_K, np.mean(hits)))

# ============================================================
# CELL 5 — Load the generator LLM + define prompts
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-3B-Instruct"   # if this is too slow on your Colab tier, use "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()

SYS = "You are a careful biomedical expert."
FMT = "Reply in exactly this format:\nANSWER: <yes|no|maybe>\nRATIONALE: <one sentence>"

def P_LLM(q):
    return f"Question: {q}\n\n{FMT}"

def P_RAG(q, P, abstain=False):
    ctx = "\n".join(f"[{n+1}] {p}" for n, p in enumerate(P))
    fmt = FMT.replace("<yes|no|maybe>", "<yes|no|maybe|insufficient>") if abstain else FMT
    extra = " If the context does not contain enough evidence, use ANSWER: insufficient." if abstain else ""
    return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer using only the context.{extra}\n{fmt}"

@torch.no_grad()
def generate(prompt, max_new_tokens=80):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(llm.device)
    out = llm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

def parse(txt):
    a = re.search(r"ANSWER:\s*(yes|no|maybe|insufficient)", txt, re.I)
    r = re.search(r"RATIONALE:\s*(.+)", txt, re.I | re.S)
    return (a.group(1).lower() if a else "invalid"), (r.group(1).strip().split("\n")[0] if r else "")

# ============================================================
# CELL 6 — Run generations for all 3 systems (B1, B2, B3)
# CHECKPOINTED: safe to stop and re-run this cell; it resumes.
# Takes roughly 45-90 minutes on a free T4 for 300 questions.
# ============================================================
RES = f"{OUT}/generations.json"
results = json.load(open(RES)) if os.path.exists(RES) else {}

for i in tqdm(dev_ids + test_ids):
    if str(i) in results:
        continue
    q = ds[i]["question"]
    P, S, SRC = retrieve(q)
    results[str(i)] = dict(
        q=q, gold=ds[i]["final_decision"], status=status[i],
        passages=P, sims=S, hit=int(i in SRC),
        b1=generate(P_LLM(q)),
        b2=generate(P_RAG(q, P)),
        b3=generate(P_RAG(q, P, abstain=True)),
    )
    if len(results) % 20 == 0:
        json.dump(results, open(RES, "w"))

json.dump(results, open(RES, "w"))
print("Done. Total cached:", len(results))

# ============================================================
# CELL 7 — Verifier (inside method) + independent judge (evaluation only)
# ============================================================
from transformers import AutoModelForSequenceClassification

def load_nli(name):
    t = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
    ent = [i for i, l in m.config.id2label.items() if l.lower().startswith("entail")][0]
    return t, m, ent

VER = load_nli("cross-encoder/nli-deberta-v3-base")              # used inside the method
JUD = load_nli("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")   # used only to score hallucination

@torch.no_grad()
def entail(nli, passages, hyp):
    t, m, ent = nli
    enc = t(passages, [hyp] * len(passages), truncation=True, max_length=512,
            padding=True, return_tensors="pt").to("cuda")
    return torch.softmax(m(**enc).logits, -1)[:, ent].tolist()

rows = []
for k, r in tqdm(results.items()):
    row = dict(pubid=int(k), split="dev" if int(k) in dev_set else "test",
               status=r["status"], gold=r["gold"], hit=r["hit"], sim=max(r["sims"]))
    for s in ("b1", "b2", "b3"):
        a, rat = parse(r[s])
        row[f"{s}_ans"] = a
        row[f"{s}_faith"] = max(entail(JUD, r["passages"], rat)) if rat else 0.0
    _, rat = parse(r["b2"])
    pv = entail(VER, r["passages"], rat) if rat else [0.0] * len(r["passages"])
    row["v1"], row["v3"] = pv[0], max(pv)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/scored.csv", index=False)
print(df.groupby(["split", "status"]).size())

# ============================================================
# CELL 8 — Metrics, threshold tuning on dev, main results table on test
# ============================================================
def metrics(d, answered, ans, faith, lam=1.0):
    A = (d.status == "answerable").values
    U = ~A
    answered = np.asarray(answered, bool)
    ans = np.asarray(ans)
    gold = d.gold.values
    faith = np.asarray(faith) >= 0.5
    good = answered & (ans == gold) & A
    bad = answered & ~good
    return dict(
        Acc_A=good.sum() / A.sum(),
        Cov_A=answered[A].mean(),
        SelAcc=good.sum() / max(answered[A].sum(), 1),
        UAR=answered[U].mean(),
        AbstPrec=(~answered & U).sum() / max((~answered).sum(), 1),
        Halluc=1 - faith[answered].mean() if answered.any() else 0.0,
        Utility=(good.sum() - lam * bad.sum()) / len(d),
    )

def get(d, name, tau=None):
    T = np.ones(len(d), bool)
    if name == "B1": return T, d.b1_ans, d.b1_faith
    if name == "B2": return T, d.b2_ans, d.b2_faith
    if name == "B3": return (d.b3_ans != "insufficient").values, d.b3_ans, d.b3_faith
    if name == "Gate-sim":  return (d.sim >= tau).values, d.b2_ans, d.b2_faith
    if name == "Ours-top1": return (d.v1 >= tau).values,  d.b2_ans, d.b2_faith
    if name == "Ours":      return (d.v3 >= tau).values,  d.b2_ans, d.b2_faith

def tune(dev, name, grid, lam=1.0):
    return max(grid, key=lambda t: metrics(dev, *get(dev, name, t), lam)["Utility"])

dev  = df[df.split == "dev"].reset_index(drop=True)
test = df[df.split == "test"].reset_index(drop=True)

grid  = np.linspace(0, 1, 101)
sgrid = np.linspace(df.sim.min(), df.sim.max(), 101)
tau = {n: tune(dev, n, sgrid if n == "Gate-sim" else grid) for n in ["Gate-sim", "Ours-top1", "Ours"]}
print("Tuned thresholds (dev):", tau)

order = ["B1", "B2", "B3", "Gate-sim", "Ours-top1", "Ours"]
table = pd.DataFrame({n: metrics(test, *get(test, n, tau.get(n))) for n in order}).T.round(3)
print(table)
table.to_csv(f"{OUT}/main_table.csv")

# ============================================================
# CELL 9 — Bootstrap CIs, lambda sweep, trade-off figure, human-check sample
# ============================================================
def boot(name, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(test)
    out = []
    for _ in range(B):
        d = test.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        out.append(metrics(d, *get(d, name, tau.get(name))))
    o = pd.DataFrame(out)
    return pd.DataFrame({"lo": o.quantile(.025), "hi": o.quantile(.975)}).round(3)

print("Ours 95% CI:\n", boot("Ours"))
print("B2 95% CI:\n", boot("B2"))

lam_tab = {}
for lam in [0.25, 0.5, 1, 2, 4]:
    t = tune(dev, "Ours", grid, lam)
    lam_tab[lam] = {
        "B2": metrics(test, *get(test, "B2"), lam)["Utility"],
        "B3": metrics(test, *get(test, "B3"), lam)["Utility"],
        "Ours": metrics(test, *get(test, "Ours", t), lam)["Utility"],
        "tau": t,
    }
print(pd.DataFrame(lam_tab).T.round(3))

import matplotlib.pyplot as plt

def curve(name, g):
    pts = [(m["UAR"], m["Cov_A"]) for m in (metrics(test, *get(test, name, t)) for t in g)]
    return zip(*pts)

plt.figure(figsize=(3.4, 2.8))
x, y = curve("Ours", grid);      plt.plot(x, y, label="NLI verifier (ours)")
x, y = curve("Gate-sim", sgrid); plt.plot(x, y, "--", label="Retrieval-sim gate")
for n, mk in [("B2", "s"), ("B3", "^")]:
    m = metrics(test, *get(test, n))
    plt.scatter(m["UAR"], m["Cov_A"], marker=mk, label=n)
plt.xlabel("UAR (lower is better)")
plt.ylabel("Coverage on answerable")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig(f"{OUT}/tradeoff.pdf")
print("Saved figure to", f"{OUT}/tradeoff.pdf")

# 50-row sample for a manual faithfulness check (strengthens the paper)
sample = test.sample(50, random_state=0)[["pubid", "b2_ans", "b2_faith"]]
sample.to_csv(f"{OUT}/manual_check.csv", index=False)
print("Saved manual-check sample to", f"{OUT}/manual_check.csv")
print("\nALL DONE. Files in", OUT, ":", os.listdir(OUT))

# ============================================================
# CELL 10 — Verifier diagnostics (AUC, score histogram) + multi-lambda tables
# Produces verifier_separation.pdf (Fig. 2) and main_table_lambda*.csv
# ============================================================
from sklearn.metrics import roc_auc_score

test['should_answer'] = (test.status == 'answerable') & (test.b2_ans == test.gold)
auc = roc_auc_score(test.should_answer, test.v3)
print("AUC of verifier score (v3) for 'should answer':", round(auc, 3))

plt.figure(figsize=(4, 3))
plt.hist(test.v3[test.should_answer], bins=20, alpha=0.6, label='correct & answerable')
plt.hist(test.v3[~test.should_answer], bins=20, alpha=0.6, label='wrong or unanswerable')
plt.xlabel('verifier score (v3)'); plt.ylabel('count'); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/verifier_separation.pdf")

for lam_report in [0.25, 0.5, 1.0]:
    tau_r = {n: tune(dev, n, sgrid if n == "Gate-sim" else grid, lam_report)
             for n in ["Gate-sim", "Ours-top1", "Ours"]}
    t2 = pd.DataFrame({n: metrics(test, *get(test, n, tau_r.get(n)), lam_report)
                       for n in order}).T.round(3)
    print(f"\n=== Main table at lambda={lam_report} ===")
    print(t2)
    t2.to_csv(f"{OUT}/main_table_lambda{lam_report}.csv")
