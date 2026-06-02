# Claude Code Task

Objective:

Implement a DSD Runtime PoC.

Model:

Gemma 3 12B

Requirements:

1. Load model
2. Obtain final hidden state
3. Compute logits incrementally along hidden dimensions
4. Maintain:

Δ = top1 - top2

B = remaining upper bound

5. Stop when:

Δ > B

Outputs:

- predicted token
- dense token
- stop_dim
- skip_rate

Constraints:

- correctness first
- no CUDA optimization
- no throughput optimization
- exact Top1 matching required
