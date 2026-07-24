# Deep Learning with PyTorch — Deep-Dive Track

Deep learning is where "the model" stops being a library call and becomes a system you design, train, debug, and operate. A senior AI engineer must be able to derive a backward pass on a whiteboard, write a production training loop from memory, explain why a loss curve looks the way it does, and diagnose NaNs, silent data bugs, and GPU starvation without guessing. The original single-file guide for this phase compressed all of that into an outline; this folder expands it into six deep sub-guides, each with line-by-line math derivations (shapes annotated), complete runnable code with expected output, incident-style war stories, and senior-level interview drills.

Part of the [Senior AI Engineer Roadmap](../00-Senior-AI-Engineer-Roadmap.md) — Phase 3.

---

## The Sub-Guides

| # | Guide | What you will be able to do afterwards |
| --- | --- | --- |
| 1 | [Neural Networks from Scratch](./01-Neural-Networks-from-Scratch.md) | Derive forward pass, every loss function (MSE, BCE, categorical CE with the softmax gradient worked to `y_hat − y`), and full 2-layer backprop; implement a modular NumPy network with gradient checking and train it on a real problem. |
| 2 | [Training Dynamics](./02-Training-Dynamics.md) | Derive Xavier/He initialization, explain BatchNorm vs LayerNorm at depth (running stats, small-batch failure), pick activations/regularizers/LR schedules deliberately, and read any loss curve like an ECG. |
| 3 | [PyTorch Mastery](./03-PyTorch-Mastery.md) | Explain tensor storage/views/contiguity, autograd graph mechanics, `nn.Module` internals, DataLoader pitfalls; write the complete production training loop (AMP, accumulation, full-resume checkpointing, early stopping, logging) and profile it. |
| 4 | [CNNs and Computer Vision](./04-CNNs-and-Computer-Vision.md) | Compute conv output sizes, parameter counts, and receptive fields by hand; explain VGG→ResNet→ViT; fine-tune pretrained backbones properly; implement IoU/NMS and explain mAP precisely; spec the defect-detection capstone. |
| 5 | [Sequence Models and Forecasting](./05-Sequence-Models-and-Forecasting.md) | Derive BPTT and the vanishing-gradient product, derive LSTM gates equation by equation, and do honest forecasting engineering: rolling-origin backtests, baselines that beat deep models, quantile/conformal intervals, metric pitfalls. |
| 6 | [Debugging Neural Networks](./06-Debugging-Neural-Networks.md) | Run the systematic debugging playbook: overfit-one-batch, input-pipeline verification, the loss-not-decreasing decision tree, NaN hunting, a gallery of silent bugs with fixes, reproducibility, and performance debugging. |

## Suggested Order

1. **01 → 02** first: the from-scratch derivations (01) make every concept in training dynamics (02) concrete rather than folklore.
2. **03** next: with the math internalized, PyTorch's machinery (autograd, Modules, the production loop) maps onto things you have already built by hand.
3. **04 or 05** depending on your target domain — vision (04) and sequences/forecasting (05) are independent of each other; do both before interviews.
4. **06 last, then forever**: the debugging playbook assumes vocabulary from all previous guides, and it is the guide you will reopen most often on the job.

Each guide is self-contained enough to revisit individually. Every guide ends with **Production War Stories**, **Best Practices**, and **Interview Drills** — do the drills cold (answer out loud before expanding) as interview rehearsal.

## Relationship to the Rest of the Track

- The original single-file overview remains at [`../06-Deep-Learning-with-PyTorch.md`](../06-Deep-Learning-with-PyTorch.md) as a quick-review summary; this folder is the deep version.
- Prerequisites: [Mathematics for AI](../02-Mathematics-for-AI.md) (linear algebra, calculus/chain rule) and [Classical Machine Learning](../05-Classical-Machine-Learning.md) (bias/variance, evaluation discipline).
- Next phase: [NLP and Transformers](../07-NLP-and-Transformers.md) — the seq2seq + attention section of guide 05 is the on-ramp.

Back to the [AI Engineering track index](../README.md).
