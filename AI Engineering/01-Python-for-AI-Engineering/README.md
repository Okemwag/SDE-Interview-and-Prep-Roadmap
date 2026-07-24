# Python for AI Engineering — Deep Track

Python is the lingua franca of AI, but the Python that ships production ML systems looks nothing like notebook-tutorial Python. This track expands the original single-file guide into six deep sub-guides covering the language internals, concurrency machinery, scientific stack, production tooling, service engineering, and background-job architecture that senior AI engineers are expected to know cold — with runnable code, real failure modes, and interview drills in every guide.

The audience is an experienced backend engineer who knows another language deeply and Python moderately: the guides skip syntax tutorials and go straight to mechanisms, trade-offs, and the things that break in production.

Part of the [Senior AI Engineer Roadmap](../00-Senior-AI-Engineer-Roadmap.md) — Phase 1.

## Sub-guides

| # | Guide | What it covers |
|---|-------|----------------|
| 1 | [Advanced Python Language](./01-Advanced-Python-Language.md) | Generators and lazy pipelines, context managers, decorators, the typing system (generics, Protocols, ParamSpec), dataclasses vs Pydantic vs attrs, descriptors, `__slots__`, and the CPython memory model. |
| 2 | [Async, Concurrency, and Multiprocessing](./02-Async-Concurrency-and-Multiprocessing.md) | The GIL (truthfully), asyncio internals, threads vs processes vs async, multiprocessing start methods and the CUDA+fork hazard, and batched parallel LLM calls with retries. |
| 3 | [NumPy and the Scientific Stack](./03-NumPy-and-the-Scientific-Stack.md) | ndarray memory layout, strides, views vs copies, broadcasting derived from first principles, einsum, Pandas pitfalls, Polars lazy frames, and PyArrow zero-copy interchange. |
| 4 | [Production Python Engineering](./04-Production-Python-Engineering.md) | src layout, uv + pyproject.toml packaging, pytest for ML code, ruff/mypy configs, profiling (cProfile, py-spy, line/memory profilers), and structured logging. |
| 5 | [FastAPI and Service Engineering](./05-FastAPI-and-Service-Engineering.md) | Pydantic v2, dependency injection, async endpoints and thread-pool offloading, SSE token streaming, lifespan model loading, SQLAlchemy 2.0 async + Alembic, and testing. |
| 6 | [Task Queues and Background Jobs](./06-Task-Queues-and-Background-Jobs.md) | Why AI workloads need job systems, Celery in depth (brokers, retries, idempotency, visibility timeouts), Dramatiq, and a document-processing pipeline with checkpointing and DLQs. |

## Suggested order and prerequisites

Read the guides in numerical order — each builds on the previous:

1. **Guides 1–2** are the language foundation. Guide 2 assumes the generator and context-manager material from Guide 1.
2. **Guide 3** stands mostly alone but reuses the memory-model vocabulary (refcounts, buffers) from Guide 1.
3. **Guide 4** is the tooling layer everything after it assumes: its project layout, pytest, and logging patterns appear in Guides 5 and 6.
4. **Guides 5–6** are the service layer. Guide 5 assumes asyncio (Guide 2) and Pydantic (Guide 1); Guide 6 assumes Guide 5's service context — its pipeline example is the kind of work you offload *from* a FastAPI service.

Prerequisites: comfort with Python syntax, functions, classes, and virtual environments; production experience in any backend language. No prior asyncio, NumPy, or Celery experience is assumed.

The original condensed overview remains at [../01-Python-for-AI-Engineering.md](../01-Python-for-AI-Engineering.md) for a quick refresher.

Back to the [AI Engineering track index](../README.md).
