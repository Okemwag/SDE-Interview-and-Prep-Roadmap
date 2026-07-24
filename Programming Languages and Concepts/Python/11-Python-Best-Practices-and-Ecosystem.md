# Python Best Practices and Ecosystem

Writing Python that *works* is easy; writing Python that reads like Python — idiomatic, tooled, packaged, and fast enough — is what interviews for real jobs assess. This file covers PEP 8 and the Zen in practice, project structure, the modern toolchain (uv, ruff, poetry), packaging, where Python dominates industrially, and a pragmatic performance playbook.

## PEP 8 and Idiomatic Python

PEP 8 is the style guide: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants, 4-space indents, imports grouped (stdlib / third-party / local). But idiom goes deeper than style — the Zen of Python (`import this`) applied:

```python
# "Explicit is better than implicit" / "Readability counts"

# Unpythonic (C-style index loops):
for i in range(len(items)):
    print(i, items[i])

# Pythonic:
for i, item in enumerate(items):
    print(i, item)

for name, score in zip(names, scores):        # parallel iteration
    print(name, score)

# Swap, unpack, and multiple return -- no temp variables
a, b = b, a
first, *middle, last = [1, 2, 3, 4, 5]

# "Flat is better than nested": guard clauses over arrow code
def ship(order):
    if order is None:
        raise ValueError("no order")
    if not order.paid:
        return Status.AWAITING_PAYMENT
    if not order.in_stock:
        return Status.BACKORDERED
    return dispatch(order)                     # happy path at zero indent

# "There should be one obvious way": use the stdlib
total = sum(x.price for x in cart)             # not a manual accumulator loop
path = pathlib.Path(base) / "logs" / "app.log" # not string concatenation
text = ", ".join(words)                        # not += in a loop (O(n^2))
```

Don't fight duck typing with `isinstance` ladders; don't write getters/setters (use properties); don't compare booleans (`if flag:` not `if flag == True:`).

## Project Structure

The modern convention is the **src layout**, which prevents accidentally importing uninstalled local code and forces tests to run against the installed package:

```text
myproject/
├── pyproject.toml          # single config file: metadata, deps, tools
├── README.md
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── api/            # e.g. FastAPI routers
│       ├── core/           # domain logic (framework-free)
│       └── db/
├── tests/
│   ├── conftest.py
│   └── test_core.py
└── .github/workflows/ci.yml
```

`pyproject.toml` (PEP 517/518/621) replaced `setup.py` + `setup.cfg` + `requirements.txt` sprawl as the one place for metadata, dependencies, and tool config (ruff, mypy, pytest all read it).

## Virtual Environments and Modern Tooling

Every project gets an isolated environment — never install into the system Python.

```bash
# Classic stdlib way
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Modern way: uv (Rust-based, 10-100x faster, all-in-one)
uv init myproject          # scaffold with pyproject.toml
uv add fastapi pytest      # resolve + install + lock (uv.lock)
uv run pytest              # run inside the env, no activation needed
uv python install 3.13     # even manages Python versions themselves
```

The 2025-era toolbox worth naming in interviews:

| Tool | Role |
|---|---|
| **uv** | Package/env/Python-version manager; extremely fast; lockfiles; largely replacing pip+venv+pyenv+pipx |
| **ruff** | Linter *and* formatter (Rust); replaces flake8 + isort + (mostly) black; `ruff check --fix`, `ruff format` |
| **poetry** | Older all-in-one dependency manager with lockfiles; still common |
| **pip + venv** | The stdlib baseline; fine for simple projects |
| **mypy / pyright** | Static type checking |
| **pre-commit** | Runs lint/format/type hooks on every commit |

```mermaid
flowchart LR
    A["write code"] --> B["ruff format + ruff check (pre-commit)"]
    B --> C["mypy / pyright"]
    C --> D["pytest + coverage"]
    D --> E["CI (GitHub Actions)"]
    E --> F["build wheel + publish / deploy"]
```

## Packaging Basics

```toml
# pyproject.toml (PEP 621 metadata)
[project]
name = "myapp"
version = "1.2.0"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27", "pydantic>=2"]

[project.scripts]
myapp = "myapp.cli:main"        # installs a `myapp` console command

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Concepts to know: **wheel** (`.whl`, prebuilt zip — what pip prefers) vs **sdist** (source archive built on install); build with `uv build` or `python -m build`; publish to PyPI with `uv publish`/`twine`. Applications additionally pin exact versions via a lockfile (`uv.lock`, `poetry.lock`) for reproducible deploys; libraries declare *ranges* and leave pinning to the app.

## Where Python Dominates

- **Data science & analytics** — pandas/polars, NumPy, Jupyter; the lingua franca of analysis.
- **Machine learning / AI** — PyTorch, scikit-learn, Hugging Face; virtually all LLM tooling is Python-first.
- **Backend web** — **FastAPI** (async, type-hint-driven validation and OpenAPI docs) and **Django** (batteries-included ORM/admin/auth); Flask for microservices.
- **Scripting & automation** — glue code, CLIs (argparse/typer/click), DevOps tooling, CI scripts.
- **Scientific computing** — SciPy, astropy, bioinformatics pipelines.

Knowing *why* helps in interviews: readability lowers collaboration cost, the C-extension ecosystem gives near-native speed where it matters, and the package ecosystem means almost everything has a mature library.

## Performance Tips

The golden rule: **measure first**.

```python
# 1. Profile before optimizing
python -m cProfile -s cumulative app.py        # where does time go?
# py-spy top --pid 1234                        # sampling profiler, prod-safe
# python -m timeit "'-'.join(map(str, range(100)))"   # micro-benchmarks

# 2. Fix algorithms before micro-optimizing: O(n^2) -> O(n) beats any tweak
seen = set()                     # membership: set O(1) vs list O(n)

# 3. Vectorize numeric work -- drop the interpreter loop entirely
import numpy as np
a = np.arange(1_000_000)
total = (a * a).sum()            # ~100x faster than a Python-level loop

# 4. Idiomatic wins: join over +=, comprehensions over append loops,
#    generators to avoid materializing, caching with functools.lru_cache
```

When Python-level optimization runs out, the escalation ladder is:

1. **Better algorithm / data structure** (always first).
2. **Vectorization** (NumPy, pandas/polars) or a faster library (orjson, uvloop).
3. **Caching** and doing less work (memoize, batch I/O, precompute).
4. **Concurrency**: asyncio/threads for I/O, processes for CPU (see the concurrency guide).
5. **Native code**: Cython, mypyc, Rust via **PyO3/maturin** (the modern favorite — how pydantic v2 and ruff get their speed), or C extensions; Numba for JIT-compiling numeric hot loops; PyPy for pure-Python workloads.

Real-world sense of scale: a FastAPI endpoint is rarely CPU-bound — the wins are usually N+1 query elimination and caching, not language speed. Conversely, an ML feature pipeline crunching arrays should be NumPy/polars end to end, with pure-Python loops treated as a smell.

## Best Practices

- Automate style: ruff format + ruff check in pre-commit and CI; never argue about formatting in review.
- One virtual environment per project, a lockfile for every application, `requires-python` declared, and dependencies added deliberately (each one is a supply-chain and maintenance cost).
- Use the src layout with `pyproject.toml` as the single source of configuration.
- Keep domain logic framework-free (plain modules/classes); import the framework at the edges — it makes code testable and portable.
- Type-check and test in CI from day one; a green pipeline is the definition of done.
- Profile with cProfile/py-spy before optimizing; state complexity in Big-O when discussing performance.
- Prefer boring stdlib solutions (pathlib, dataclasses, itertools) over dependencies; prefer well-maintained libraries over clever hand-rolled code.
- Read `import this` once a year — and actually apply "errors should never pass silently" and "now is better than never."

## Interview Questions

<details>
<summary>What is the Zen of Python, and give two concrete examples of applying it.</summary>
Nineteen aphorisms (<code>import this</code>) capturing Python's design philosophy. Applications: "Flat is better than nested" → replace arrow-shaped if/else pyramids with early-return guard clauses; "Explicit is better than implicit" → <code>from module import name</code> over wildcard imports, keyword arguments for booleans (<code>retry(attempts=3)</code>); "Errors should never pass silently" → no <code>except: pass</code>; "There should be one obvious way" → <code>enumerate</code>/<code>zip</code>/<code>join</code> instead of index arithmetic and string concatenation loops.
</details>

<details>
<summary>Why use virtual environments, and what do uv/poetry add over pip?</summary>
A venv isolates each project's dependencies from the system and from other projects, preventing version conflicts and making builds reproducible. Plain pip installs whatever resolves <em>today</em>; uv and poetry add deterministic lockfiles (exact pinned graph, hash-verified), dependency groups (dev/test), and project workflows (<code>uv run</code>, publishing). uv additionally manages Python interpreter versions and is dramatically faster, effectively bundling pip, venv, pyenv, and pipx into one tool.
</details>

<details>
<summary>What's the difference between a wheel and an sdist?</summary>
An sdist is a source archive; installing it may execute a build backend on the user's machine (needing compilers for C extensions). A wheel (<code>.whl</code>) is a prebuilt binary/zip that pip just unpacks — faster, no toolchain needed, and tagged per platform/Python version for native code (e.g. <code>manylinux</code> wheels). Maintainers publish both; pip prefers a compatible wheel. Pure-Python projects ship one universal wheel.
</details>

<details>
<summary>Why do applications pin exact dependency versions while libraries declare ranges?</summary>
An application controls its own deployment: pinning the full resolved graph in a lockfile guarantees every environment (dev, CI, prod) runs identical code — reproducibility and safe rollbacks. A library is <em>combined</em> with other libraries inside someone else's app; if it pinned exact versions it would cause unsolvable conflicts, so it declares compatible ranges (<code>httpx>=0.27,<1</code>) and lets the application's resolver+lockfile do the final pinning.
</details>

<details>
<summary>A Python service endpoint is slow. Walk through your optimization process.</summary>
(1) Measure: APM traces or py-spy/cProfile to find where time actually goes — usually I/O. (2) If it's the database: fix N+1 queries, add indexes, batch. (3) Cache expensive/repeated work (lru_cache, Redis) with sensible invalidation. (4) If it's Python CPU: better algorithm/data structure first, then vectorize or move to a native library; consider offloading to a background worker (celery) if the response doesn't need it. (5) Concurrency: async or thread pool so waits overlap. (6) Re-measure and stop when the SLO is met — never optimize unprofiled code.
</details>

<details>
<summary>When is it justified to drop into C/Rust extensions, and what are the modern options?</summary>
Only after profiling shows a genuine CPU-bound hot loop that algorithms, vectorization, and caching can't fix — or when you need to wrap an existing native library. Options: Cython (annotate Python into C), mypyc (compile type-hinted Python), Numba (JIT for numeric functions), and Rust via PyO3/maturin — the current favorite for safety and tooling, used by pydantic-core, ruff, and polars. Costs: build complexity, per-platform wheels, harder debugging, and GIL-interaction considerations — so keep the native surface small.
</details>

<details>
<summary>What does idiomatic string building look like, and why is <code>+=</code> in a loop bad?</summary>
Strings are immutable, so each <code>s += part</code> allocates a new string and copies everything so far — O(n²) total. Idiomatic: accumulate parts in a list and <code>"".join(parts)</code>, or use a generator directly: <code>", ".join(str(x) for x in items)</code> — O(n). For building large texts incrementally, <code>io.StringIO</code> also works. (CPython has a refcount==1 in-place optimization that sometimes hides the cost, but it's an implementation detail you must not rely on.)
</details>
