from pathlib import Path
import pandas as pd
from collections import Counter

model = {
    "Salesforce/SFR-Embedding-Mistral": "1_sfr",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": "2_llm2vec",
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct": "3_qwen",
}

dataset = {
    "cora": "1_cora",
    "pubmed": "2_pubmed",
    "ogbn-arxiv": "3_ognb-arxiv",
    "ogbn-products": "4_ogbn-products",
    "arxiv_2023": "5_arxiv_2023",
}

records = []

for result in Path("results").glob('**/*.jsonl'):
    print("reading", result)
    df = pd.read_json(result, lines=True).sort_values(by='test_acc', ascending=False)
    counter = Counter()
    for x in df["gnn"]:
        counter[x] += 1
        if counter[x] >= 4:
            break
    winning_model = x

    df = df[df["gnn"] == winning_model].head(4)
    print(df)
    acc = df["test_acc"]
    avg, *_ = acc.mean(),
    pm = (acc.max() - acc.min()) / 2
    info = df.head(1).to_dict("records").pop()
    record = dict(
        dataset=dataset[info["dataset"]],
        lm=model[info["lm"]],
        model=info["gnn"],
        avg="~" + str(round(avg, 4)),
        pm="~" + str(round(pm, 4)),
    )
    print(record)
    records.append(record)
    print("done")

pd.DataFrame \
    .from_records(records) \
    .sort_values(by=["dataset", "lm"]) \
    .to_csv("results.csv", index=False)