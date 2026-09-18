"""Compare chunking configurations on the same corpus with the same questions.

Retrieval and generation are measured separately, because the same abstention has opposite fixes
depending on which produced it: a retrieval miss means the index, a generation miss means the
prompt or the acceptance rules. Each configuration is a Bedrock data source over an identical copy
of the active generation, distinguished only by the generation marker in its sidecars, so the
production runtime's generation filter never sees the experiment.

Usage: uv run --with boto3==1.40.21 python tools/chunking_experiment.py

The end-to-end answer comparison (14/18 hierarchical, 12/18 fixed-300, 11/18 semantic) was run
interactively against the same probe; its numbers are recorded in retrieval-config.yaml under
migration.measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import boto3

KB = "ONVASJDDNX"
PROFILE = "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-fable-5"
ACTIVE = "sha256:14a4618de0bc6bb559146a4ec584214f64d1667955bd14fbe29dc735a7eeced1"

# Each entry is a generation marker to filter on. The two experiment markers used on 2026-09-18
# were removed with their data sources once hierarchical was adopted; add a marker here to run a
# new comparison against an isolated copy of the active generation.
CONFIGS = {
    "production": ACTIVE,
}

# The probe: six topics, three phrasings each, the same set the 88% baseline came from. Expected
# paths are substrings any correct top-5 should contain, chosen by reading the corpus, not the
# results. A retrieval hit means at least one expected path appears in the top 5.
PROBE: list[tuple[str, str, tuple[str, ...]]] = [
    ("tsc", "Who leads the TSC?", ("MAINTAINERS.md", "GOVERNANCE.md")),
    (
        "tsc",
        "who's in charge of the valkey technical steering committee",
        ("MAINTAINERS.md", "GOVERNANCE.md"),
    ),
    ("tsc", "tsc chair?", ("MAINTAINERS.md", "GOVERNANCE.md")),
    (
        "compression",
        "How does Valkey replication compression work?",
        ("compression", "replication.c", "valkey.conf"),
    ),
    ("compression", "explain repl-compression", ("compression", "replication.c", "valkey.conf")),
    ("compression", "how do replicas negotiate lz4", ("compression", "replication.c")),
    ("contributing", "How do I contribute to Valkey?", ("CONTRIBUTING", "GOVERNANCE")),
    ("contributing", "how to submit a patch to valkey", ("CONTRIBUTING",)),
    ("contributing", "first PR to valkey, what do I need to know", ("CONTRIBUTING",)),
    (
        "failover",
        "How does Valkey failover work?",
        ("cluster", "sentinel", "failover", "replication"),
    ),
    ("failover", "what happens when a primary dies in cluster mode", ("cluster",)),
    ("failover", "explain automatic failover", ("cluster", "sentinel", "failover")),
    ("commands", "How do I use GET?", ("get.md", "t_string.c", "get.json")),
    ("commands", "what does SET do", ("set.md", "t_string.c", "set.json")),
    ("commands", "how does EXPIRE work", ("expire", "expire.c")),
    ("persistence", "how does RDB snapshotting work", ("rdb", "persistence")),
    ("persistence", "what is AOF", ("aof", "persistence")),
    ("persistence", "difference between rdb and aof", ("persistence", "rdb", "aof")),
]


def retrieve(client, question: str, marker: str, k: int = 5) -> list[dict]:
    r = client.retrieve(
        knowledgeBaseId=KB,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": k,
                "filter": {"equals": {"key": "generation_id", "value": marker}},
            }
        },
    )
    return r.get("retrievalResults", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/chunking-results.json"))
    args = parser.parse_args()

    rt = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    results: dict[str, dict] = {}
    for name, marker in CONFIGS.items():
        hits, scores, distinct, per_q = 0, [], [], []
        for topic, question, expected in PROBE:
            got = retrieve(rt, question, marker)
            paths = [g.get("metadata", {}).get("path", "") for g in got]
            hit = any(any(e.lower() in p.lower() for e in expected) for p in paths)
            top = got[0]["score"] if got else 0.0
            hits += hit
            scores.append(top)
            distinct.append(len({g.get("metadata", {}).get("document_id") for g in got}))
            per_q.append(
                {
                    "topic": topic,
                    "question": question,
                    "hit": hit,
                    "top_score": round(top, 3),
                    "paths": paths[:3],
                    "chunk_bytes": [len(g["content"]["text"]) for g in got],
                }
            )
        results[name] = {
            "retrieval_hits": hits,
            "mean_top_score": round(statistics.mean(scores), 3),
            "mean_distinct_docs_in_top5": round(statistics.mean(distinct), 2),
            "mean_chunk_bytes": round(statistics.mean(b for q in per_q for b in q["chunk_bytes"])),
            "questions": per_q,
        }
        summary = results[name]
        print(
            f"{name:26} retrieval hits {hits}/{len(PROBE)}  "
            f"mean top score {summary['mean_top_score']}  "
            f"distinct docs/top5 {summary['mean_distinct_docs_in_top5']}  "
            f"mean chunk {summary['mean_chunk_bytes']}B"
        )

    args.out.write_text(json.dumps(results, indent=1))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
