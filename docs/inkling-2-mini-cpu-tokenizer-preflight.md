# Inkling 2 Mini CPU tokenizer preflight

The future smoke harness pin is `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`. At that immutable public Hub revision, the revision API and the required `config.json`, `tokenizer_config.json`, and `tokenizer.json` endpoints resolved successfully. `AutoTokenizer.from_pretrained` succeeded with `trust_remote_code=False` using the harness's pinned Transformers source revision `6173a2f679ff4ed393afc499aefc569da7e68bc1`.

On CPU, Kev's five delimiters were added with distinct IDs (`<|fim_prefix|>` 200058, `<|fim_middle|>` 200059, `<|box_start|>` 200060, `<|box_end|>` 200061, `<|fim_suffix|>` 200062). Kev's encoder produced a 35-token representation of one benign question with two options. This check loaded tokenizer/configuration assets only; it did not load model weights or score the example.

This establishes that the selected revision is publicly resolvable and its tokenizer can be used by the pinned Transformers runtime for Kev delimiter addition and benign-text encoding. It does not establish released-weight loading, conversion correctness, actual model embedding resize behavior, memory fit, CUDA execution, latency, decision quality, or that Inkling is suitable as Kev's backbone. The separate tiny synthetic CPU fixture is not a test of the released checkpoint.

The historical H200 attempt remains recorded against its original pin `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`; it failed during tokenizer initialization before weight loading or scoring. Its report and durable marker are unchanged, and there has been no retry or further GPU allocation.
