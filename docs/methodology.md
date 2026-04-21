# Methodology

This project studies Countdown solving under cross-tokenizer distillation constraints.

Main ideas:

- teacher from a different model family
- narrow target format
- rejection sampling plus strict verifier
- warm-start before any on-policy refinement
- careful tracking of syntax-valid, numbers-valid, and exact-target-hit metrics
