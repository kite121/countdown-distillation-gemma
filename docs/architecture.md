# Architecture

The intended production-like flow is:

1. build a verified teacher corpus
2. canonicalize valid equations
3. warm-start `google/gemma-3-1b-it`
4. optionally run GOLD-style on-policy refinement
5. perform multi-sample inference with strict validation

The final deployed behavior should depend on the model itself, while external code is limited to validation and candidate selection.
