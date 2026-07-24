# Retrieval-Augmented Generation — Deep Track

RAG is how you make an LLM answer from *your* data: retrieve the relevant documents at question time and place them in the context window with instructions to answer from them. It sounds like one retrieval call; in production it is a twelve-stage pipeline, and quality problems can originate in any stage — parsing, chunking, embedding, retrieval, reranking, context construction, or generation. The senior skill this track builds is treating RAG as a system of **independently testable layers**, each with its own instrumentation, its own metrics, and its own characteristic failure modes.

This folder replaces the single-file overview with six deep sub-guides. Each one contains complete runnable Python and PostgreSQL code, derived formulas where the math matters (BM25, RRF, cosine), incident-style war stories, best practices, and senior-level interview drills with follow-up chains.

Part of the [Senior AI Engineer Roadmap](../00-Senior-AI-Engineer-Roadmap.md) — Phase 5.

---

## The Sub-Guides

| # | Guide | What it covers |
| --- | --- | --- |
| 1 | [Document Processing and Chunking](./01-Document-Processing-and-Chunking.md) | PDF/OCR extraction realities, cleaning and dedup (MinHash), fixed/recursive/semantic/structural chunking implemented, chunk-size experiments, parent-child retrieval, metadata enrichment, versioned incremental re-indexing in SQL, checkpointed ingestion pipelines |
| 2 | [Embeddings in Depth](./02-Embeddings-in-Depth.md) | Bi-encoders and contrastive training, similarity metrics derived, dimensions and Matryoshka, embedding at 10M-chunk scale with cost arithmetic, the embedding-versioning problem and blue-green migrations, domain adaptation, a recall@k evaluation harness, query/document asymmetry, ColBERT |
| 3 | [Retrieval: Search and Hybrid](./03-Retrieval-Search-and-Hybrid.md) | BM25 derived term by term and implemented from scratch, HNSW vs IVFFlat vs flat, hybrid architectures, RRF derived and implemented, the filtered-ANN recall trap, multi-tenant enforcement, recency boosts, query rewriting/expansion/HyDE/decomposition, strategy routing |
| 4 | [pgvector in Production](./04-pgvector-in-Production.md) | Complete DDL for a multi-tenant chunk store, HNSW and IVFFlat tuning parameters explained, operator classes, reading EXPLAIN ANALYZE on vector queries, iterative scans, a full hybrid dense+FTS+RRF query in one SQL statement, honest scale limits and the migration path |
| 5 | [Reranking and Context Construction](./05-Reranking-and-Context-Construction.md) | The single-vector bottleneck, cross-encoders, two-stage retrieval with a worked latency/cost budget, LLM-as-reranker implemented, refusal thresholds, lost-in-the-middle ordering, token budgeting, context compression, citation engineering with a verification pass |
| 6 | [RAG Failure Diagnosis and Advanced Patterns](./06-RAG-Failure-Diagnosis-and-Advanced-Patterns.md) | The layer-by-layer diagnostic decision tree with exact instrumentation, the failure catalog with fixes, agentic/iterative retrieval, GraphRAG, multi-hop, RAPTOR-style summarization indexes, answer caching, freshness-critical RAG, a full-stack cost/latency budget |

## Suggested Order

Read them in numerical order — the sequence mirrors the pipeline itself:

1. **Guides 1–2** cover the ingestion path (documents → chunks → vectors). Everything downstream inherits the quality ceiling set here.
2. **Guides 3–4** cover the retrieval layer, first conceptually (BM25, ANN, hybrid, fusion) and then concretely on PostgreSQL + pgvector, the roadmap's recommended starting stack.
3. **Guide 5** covers the precision layer: reranking and assembling the final prompt context.
4. **Guide 6** ties the system together — how to diagnose a bad answer layer by layer, and the advanced patterns you reach for once the basics are solid.

If you are preparing for interviews on a deadline, guides 3, 5, and 6 carry the highest question density; if you are building a system this quarter, start at guide 1 and do not skip it — parsing and chunking failures are unrecoverable downstream.

The original single-file overview remains at [../09-Retrieval-Augmented-Generation.md](../09-Retrieval-Augmented-Generation.md) as a quick-reference summary. Evaluation methodology (gold sets, LLM judges, regression gates) is covered in guide 11 of the roadmap.

---

[Back to the AI Engineering track](../README.md)
