"""
evaluate.py - Automated Benchmark & Evaluation Suite for Epilepsy RAG
====================================================================
Tests the RAG system against 5 core clinical benchmark questions,
measuring retrieval recall, reranker confidence, latency, and answer quality.
"""

import time
import json
from rag_engine import EpilepsyRAGEngine

BENCHMARK_SUITE = [
    {
        "id": "Q1_NEUROMODULATION",
        "question": "What are the three invasive neuromodulation options approved for adult drug-resistant epilepsy, and what are their typical responder rates?",
        "expected_entities": ["VNS", "ANT-DBS", "RNS", "40%-60%", "75%"],
        "category": "Therapeutics / Devices"
    },
    {
        "id": "Q2_DRE_DEFINITION",
        "question": "How is Drug-Resistant Epilepsy (DRE) clinically defined by the ILAE consensus?",
        "expected_entities": ["failure of adequate trials", "two tolerated", "appropriately chosen", "antiepileptic drugs", "seizure freedom"],
        "category": "Clinical Diagnostics"
    },
    {
        "id": "Q3_KETOGENIC_DIET",
        "question": "What is the clinical efficacy and responder rate of Ketogenic and Modified Atkins Diet (MAD) in drug-resistant epilepsy?",
        "expected_entities": ["50%", "seizure reduction", "three times", "sixfold", "GLUT1"],
        "category": "Dietary Interventions"
    },
    {
        "id": "Q4_CIRCADIAN_PATTERNS",
        "question": "What cyclic and circadian patterns govern epileptic seizures, and why is chronotherapy relevant?",
        "expected_entities": ["circadian", "sleep/wake", "time-dependent", "chronotherapy", "seizure onset"],
        "category": "Pathophysiology"
    },
    {
        "id": "Q5_SURGERY_RESECTION",
        "question": "When is resective surgery or ablation indicated for epilepsy, and what are the main clinical considerations?",
        "expected_entities": ["presurgical evaluation", "epileptogenic zone", "eloquent cortex", "resection", "ablation"],
        "category": "Surgical Interventions"
    }
]


def run_benchmark():
    print("=" * 80)
    print("STARTING EPILEPSY CLINICAL RAG BENCHMARK EVALUATION")
    print("=" * 80)

    engine = EpilepsyRAGEngine()
    results = []

    total_latency = 0.0
    total_coverage = 0.0

    for i, test in enumerate(BENCHMARK_SUITE, start=1):
        print(f"\n[{i}/{len(BENCHMARK_SUITE)}] Category: {test['category']}")
        print(f"Question: {test['question']}")
        print("-" * 60)

        t_start = time.time()
        output = engine.ask(test["question"], top_k=6)
        elapsed = time.time() - t_start
        total_latency += elapsed

        answer = output["answer"]
        top_score = output["sources"][0]["score"] if output["sources"] else 0.0
        top_page = output["sources"][0]["page"] if output["sources"] else "N/A"

        # Check entity hit rate in answer + context
        answer_lower = answer.lower()
        matched_entities = [e for e in test["expected_entities"] if e.lower() in answer_lower]
        coverage_pct = (len(matched_entities) / len(test["expected_entities"])) * 100
        total_coverage += coverage_pct

        print(f"Latency: {elapsed:.2f}s | Top Reranker Score: {top_score:.4f} (Page {top_page})")
        print(f"Entity Keyword Coverage: {coverage_pct:.1f}% ({len(matched_entities)}/{len(test['expected_entities'])})")
        print(f"Answer Summary:\n{answer[:300]}...\n")

        results.append({
            "id": test["id"],
            "category": test["category"],
            "question": test["question"],
            "latency_seconds": round(elapsed, 2),
            "top_reranker_score": round(top_score, 4),
            "top_source_page": top_page,
            "entity_coverage_pct": round(coverage_pct, 1),
            "matched_entities": matched_entities,
            "answer": answer,
            "retrieval_queries": output.get("queries", [])
        })

    # Summary Metrics
    avg_latency = total_latency / len(BENCHMARK_SUITE)
    avg_coverage = total_coverage / len(BENCHMARK_SUITE)

    print("=" * 80)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Benchmark Queries : {len(BENCHMARK_SUITE)}")
    print(f"Average Latency         : {avg_latency:.2f} seconds")
    print(f"Average Entity Coverage : {avg_coverage:.1f}%")
    print("=" * 80)

    # Save results to JSON for presentation
    output_json_path = r"F:\Lectures\AI Hacathon\benchmark_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_queries": len(BENCHMARK_SUITE),
                "avg_latency_seconds": round(avg_latency, 2),
                "avg_entity_coverage_pct": round(avg_coverage, 1)
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed evaluation results saved to: {output_json_path}")


if __name__ == "__main__":
    run_benchmark()
