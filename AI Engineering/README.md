# AI Engineering

This folder is a complete study track for becoming a **Senior AI Engineer** — someone who can take an ambiguous business problem and design, build, evaluate, deploy, secure, scale, and continuously improve an AI system that creates measurable value.

The spine of the track is the [**Senior AI Engineer Roadmap**](./00-Senior-AI-Engineer-Roadmap.md). Every phase of that roadmap is expanded into a detailed study guide below, each with diagrams, code examples, best practices, and interview questions.

## Study Guides

### Foundations (Phase 1)

| # | Guide | Covers |
|---|-------|--------|
| 01 | [Python for AI Engineering](./01-Python-for-AI-Engineering.md) | Scientific and production Python, project structure, notebooks-to-packages discipline |
| 02 | [Mathematics for AI](./02-Mathematics-for-AI.md) | Linear algebra, calculus, and optimization — with from-scratch NumPy implementations |
| 03 | [Probability and Statistics for AI](./03-Probability-and-Statistics-for-AI.md) | Distributions, Bayes, MLE, calibration, A/B testing, why accuracy misleads |
| 04 | [SQL and Data Engineering for ML](./04-SQL-and-Data-Engineering-for-ML.md) | Point-in-time joins, leakage, training-serving skew, reproducible datasets |

### Modelling (Phases 2–4)

| # | Guide | Covers |
|---|-------|--------|
| 05 | [Classical Machine Learning](./05-Classical-Machine-Learning.md) | Supervised/unsupervised learning, evaluation, feature engineering, experiment discipline |
| 06 | [Deep Learning with PyTorch](./06-Deep-Learning-with-PyTorch.md) | Neural nets from scratch, PyTorch training loops, CNNs, time-series models |
| 07 | [NLP and Transformers](./07-NLP-and-Transformers.md) | Attention, transformer architectures, fine-tuning, PEFT/LoRA, Hugging Face |

### Generative AI Applications (Phases 5–6)

| # | Guide | Covers |
|---|-------|--------|
| 08 | [GenAI Application Engineering](./08-GenAI-Application-Engineering.md) | Model API engineering, the AI gateway pattern, structured generation |
| 09 | [Retrieval-Augmented Generation](./09-Retrieval-Augmented-Generation.md) | The full RAG pipeline, hybrid search, pgvector, reranking, citations |
| 10 | [Agentic Systems, Memory and Multimodal](./10-Agentic-Systems-Memory-and-Multimodal.md) | Agents as constrained workflows, tool authorization, memory, multimodal AI |
| 11 | [AI Evaluation Engineering](./11-AI-Evaluation-Engineering.md) | Eval datasets, RAG and agent metrics, LLM-as-judge, online evaluation |

### Production Systems (Phases 7–10)

| # | Guide | Covers |
|---|-------|--------|
| 12 | [MLOps and LLMOps](./12-MLOps-and-LLMOps.md) | Reproducibility, MLflow, training pipelines, CI/CD for AI, deployment strategies |
| 13 | [Model Serving and Inference Optimization](./13-Model-Serving-and-Inference-Optimization.md) | Serving patterns, vLLM/Ray Serve/Triton, latency percentiles, batching, KV cache |
| 14 | [GPU and Distributed AI Systems](./14-GPU-and-Distributed-AI-Systems.md) | GPU fundamentals, VRAM budgeting, DDP/FSDP, parallelism strategies |
| 15 | [Cloud, Kubernetes and Infrastructure for AI](./15-Cloud-Kubernetes-and-Infrastructure-for-AI.md) | GPU containers, K8s scheduling and autoscaling, Terraform for AI infra |

### Operating AI in Production (Phases 11–13)

| # | Guide | Covers |
|---|-------|--------|
| 16 | [AI Observability and Reliability](./16-AI-Observability-and-Reliability.md) | AI-specific telemetry, OpenTelemetry, reliability patterns, graceful degradation |
| 17 | [AI Security, Privacy and Governance](./17-AI-Security-Privacy-and-Governance.md) | OWASP LLM Top 10, prompt-injection defence, tenant isolation, responsible AI |
| 18 | [AI System Design](./18-AI-System-Design.md) | The 9-part senior design framework with fully worked case studies |

### Seniority (Phases 14–15 and beyond)

| # | Guide | Covers |
|---|-------|--------|
| 19 | [Product, Business and Leadership](./19-Product-Business-and-Leadership.md) | Metrics hierarchy, human-in-the-loop design, senior decision-making, incidents |
| 20 | [Portfolio Projects](./20-Portfolio-Projects.md) | Five production-grade portfolio projects with architectures and milestones |
| 21 | [Study Plan and Readiness Checklist](./21-Study-Plan-and-Readiness-Checklist.md) | 18-month progression, weekly structure, anti-patterns, readiness checklist |

## Suggested study order

Follow the numbering. Phases build on each other: foundations → classical ML → deep learning → transformers → generative AI applications → evaluation → production systems → security and design → senior behaviours. Do not treat them as isolated — build production projects continuously while studying theory (guide 20 tells you what to build at each stage).

## Related folders

- [Python language deep-dive](../Programming%20Languages%20and%20Concepts/Python/README.md) — the AI guides assume working Python; go here for the language itself
- [System Design](../System%20Design/) and [Databases](../Databases/) — general (non-AI) counterparts to guides 18 and 04

## Contributing

Additions are welcome — follow the existing file format: overview, Mermaid diagrams, runnable code examples, a Best Practices section, and collapsible interview questions.
