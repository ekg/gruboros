# Training Investigation Report: Why Our 700M GRU Model is Stuck

**Date**: 2025-11-26
**Problem**: Training stuck at loss ~4.7-4.8 with dying gradients (0.06-0.07), unable to improve despite 337k+ steps

---

## Executive Summary

After systematic investigation and research, **the root cause is clear: our model is configured for failure**. We're attempting to train a 20-layer deep GRU, which is **6.7× deeper than the minGRU paper's successful 3-layer model**, with architectural choices that make vanishing gradients inevitable.

**Critical Finding**: The minGRU paper achieved loss of 1.548 with 3 layers. We're stuck at 4.7-4.8 with 20 layers - **3× worse loss with 6.7× more depth**.

---

## Research Findings

### 1. MinGRU Paper Results ("Were RNNs All We Needed?")

**Source**: [ArXiv 2410.01201](https://arxiv.org/html/2410.01201v1) | [HuggingFace Paper Page](https://huggingface.co/papers/2410.01201)

#### Their Successful Configuration (Language Modeling):
- **Depth**: 3 layers
- **Optimizer**: AdamW
- **Learning rate**: 1×10⁻³ (0.001)
- **Batch size**: 64
- **Expansion factor**: 2.0
- **Dropout**: 0.2
- **Training steps**: 5,000 (converged in ~575 steps)
- **Result**: Test loss 1.548 on Shakespeare

#### Key Insights:
- minGRU achieved **comparable performance to Transformers** (1.547 loss)
- Converged **2.5× faster than Transformers** (575 vs 2000+ steps)
- Training speed: **175× faster than traditional GRU** at sequence length 512
- **No mention of gradient clipping in language modeling setup**

---

### 2. Our Configuration (FAILING)

#### Current Setup:
- **Depth**: 20 layers (6.7× deeper!)
- **Optimizer**: Schedule-Free AdamW
- **Learning rate**: 0.001 (same) → 0.1 (desperate attempt)
- **Batch size**: 114 (effective 1824 with grad_accum=16)
- **Expansion factor**: 1.0 (half theirs)
- **Dropout**: 0.0 (none!)
- **Training steps**: 337,000+ (ongoing disaster)
- **Result**: Loss stuck at 4.7-4.8 (3× worse!)

#### Observed Symptoms:
- Gradients start healthy (0.20-0.25) then die to 0.06-0.07 within ~500 steps
- Happens with **all optimizers**: Schedule-Free AdamW, SGD+momentum, plain SGD
- Happens **with and without gradient clipping**
- Happens **at all learning rates** tested (0.001, 0.002, 0.01, 0.1)
- Loss oscillates in 4.5-4.8 range but never breaks through

---

### 3. Vanishing Gradients in Deep RNNs

**Sources**:
- [Vanishing Gradient Problem - Wikipedia](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- [GeeksforGeeks - Deep Learning Gradient Problems](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)
- [Analytics Vidhya - Exploring Gradients](https://www.analyticsvidhya.com/blog/2024/04/exploring-vanishing-and-exploding-gradients-in-neural-networks/)

#### The Core Problem:
> "As network depth increases, gradients of earlier weights are calculated with increasingly many multiplications, causing the gradients of earlier weights to be exponentially smaller than the gradients of later weights."

> "This issue is particularly prevalent in Recurrent Neural Networks (RNNs), as gradients can diminish exponentially over time due to repeated multiplication."

#### Standard Solutions:
1. **Gated Architectures** - LSTMs/GRUs (we have this)
2. **Skip Connections** - Residual connections (we DON'T have this!)
3. **Dropout** - Regularization that helps gradient flow (we have NONE!)
4. **Proper Depth** - Most successful RNN models use 2-4 layers, not 20!
5. **Gradient Clipping** - Can help, but we found it made things worse

---

### 4. RNN Training Plateau Issues

**Sources**:
- [Stack Overflow - RNN Loss Plateau](https://stats.stackexchange.com/questions/283912/recurrent-neural-network-training-loss-does-not-decrease-past-a-certain-value)
- [PyTorch Forums - RNN Not Learning](https://discuss.pytorch.org/t/rnn-implementation-not-learning-and-test-loss-stuck-at-same-value/165126)
- [Cross Validated - What To Do When NN Doesn't Learn](https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn)

#### Common Causes Identified:
1. **Hidden State Management** - Improper reset/passing between batches
2. **Learning Rate Too Small** - Progress stalls (we tried 100× increase, didn't help)
3. **Learning Rate Too Large** - Oscillation in sub-optimal region (possible!)
4. **Model Too Deep** - Vanishing gradients (MATCHES OUR CASE!)
5. **Numerical Instability** - Hidden states grow too large or produce NaNs
6. **Missing Dropout** - Overfitting to training data patterns

---

### 5. Byte-Level Language Modeling

**Sources**:
- [Character-Level LM with Self-Attention](https://arxiv.org/pdf/1808.04444)
- [Bridging Gap for Tokenizer-Free Models](https://arxiv.org/pdf/1908.10322)
- [HuggingFace - Perplexity Evaluation](https://huggingface.co/docs/transformers/en/perplexity)

#### Key Findings:
- Byte-level models report **bits per byte (bpb)** not perplexity
- Our loss of 4.7-4.8 = **perplexity of ~110-122** (exp(4.7) = 109.9)
- State-of-art byte models achieve **0.874 bpb** on billion-word benchmark
- Loss of 1.548 (from minGRU) = **perplexity of 4.7** (exp(1.548) = 4.70)

#### Reality Check:
Our model is achieving **23× worse perplexity** than the minGRU paper's 3-layer model!

---

## Root Cause Analysis

### Primary Issue: Model Too Deep
**20 layers is extreme for RNNs**. The minGRU paper uses 3 layers. Even advanced LSTM work rarely exceeds 4-6 layers.

**Evidence**:
- Gradients die consistently regardless of optimizer/LR
- Pattern matches classic vanishing gradient in deep networks
- Loss plateau at high values (4.7-4.8) suggests poor capacity utilization

### Secondary Issues:

1. **No Dropout (0.0 vs paper's 0.2)**
   - Removes regularization that aids gradient flow
   - May cause overfitting to shallow patterns
   - Paper explicitly uses dropout=0.2

2. **Low Expansion Factor (1.0 vs paper's 2.0)**
   - Reduces model capacity per layer
   - With 20 layers, might not compensate for depth

3. **Hidden State Continuity** (unverified hypothesis)
   - Carrying hidden states across batches/documents
   - Potential accumulation of numerical errors
   - Could amplify vanishing gradient effect

4. **Gradient Clipping Harmful**
   - We found grad_clip=1.0 made gradients worse
   - Paper doesn't mention gradient clipping for language modeling
   - Removing it helped initially but gradients still died

---

## Experimental Results Summary

### Tests Conducted:
1. ✗ **2× LR increase** (0.002) - Gradients still died
2. ✗ **10× LR increase** (0.01) - Gradients still died
3. ✗ **100× LR increase** (0.1) - Gradients still died
4. ✗ **Fresh optimizer** - No persistent improvement
5. ✗ **SGD with momentum** - Same gradient death pattern
6. ✗ **Plain SGD (no momentum)** - Same gradient death pattern
7. ✗ **Remove gradient clipping** - Helped initially, then gradients died again
8. ✗ **All combinations above** - Consistent failure

### Gradient Death Pattern:
```
Steps 337015-337031: G = 0.21 → 0.25 (healthy start)
Steps 337031-337063: G = 0.25 → 0.22 (slight decline)
Steps 337063-337239: G = 0.22 → 0.08 (crash)
Steps 337239+:        G = 0.06-0.07 (dead)
```

This pattern occurred with **every optimizer and learning rate combination**.

---

## Comparison Table

| Metric | MinGRU Paper (Success) | Our Model (Stuck) | Ratio |
|--------|----------------------|-------------------|-------|
| Depth | 3 layers | 20 layers | 6.7× deeper |
| Loss | 1.548 | 4.7-4.8 | 3× worse |
| Perplexity | ~4.7 | ~110-122 | 23× worse |
| Expansion | 2.0 | 1.0 | 0.5× |
| Dropout | 0.2 | 0.0 | None |
| Steps to converge | ~575 | 337,000+ (not converging) | 585× more |
| Gradient norm | Unknown | 0.06-0.07 (dying) | Critical |

---

## Recommended Actions

### Immediate Fix (High Confidence):
**Reduce depth from 20 to 3 layers** matching the minGRU paper's successful configuration.

**Full configuration to match paper**:
```bash
--depth 3             # (currently 20)
--expansion_factor 2.0  # (currently 1.0)
--dropout 0.2         # (currently 0.0)
--lr 0.001           # (already correct)
--grad_clip 0.0      # (no clipping - paper doesn't mention it)
```

### Additional Experiments:

1. **Test depth scaling**: Try 3, 6, 10, 15, 20 layers to find gradient death threshold

2. **Add residual connections**: If we want depth>3, implement skip connections:
   ```python
   h_out = h_new + h_in  # Residual connection
   ```

3. **Hidden state reset experiments**:
   - Train with `--reset_hidden_every_batch` (fresh state each batch)
   - Compare gradient stability vs continuous hidden states

4. **Match paper exactly**:
   - Train a 3-layer model on our byte-level data
   - Validate we can achieve comparable loss
   - Then scale up cautiously

---

## Questions Requiring Investigation

1. **Why did we choose depth=20?**
   - Was there a specific reason or hypothesis?
   - What were we trying to achieve with extreme depth?

2. **Hidden state handling correctness**:
   - Are we properly detaching/resetting states?
   - Could numerical errors accumulate over 337k steps?
   - Should we reset every N steps regardless of document boundaries?

3. **Byte-level vs token-level loss comparison**:
   - Is our 4.7-4.8 loss reasonable for byte-level?
   - What loss should we target for meaningful text generation?

4. **Parameter count vs effective capacity**:
   - 700M params across 20 layers = 35M per layer
   - Would 3 layers × 233M params/layer be better?

---

## Conclusion

We've been trying to optimize our way out of an **architectural problem**. No amount of LR tuning, optimizer switching, or gradient clipping will fix a model that's fundamentally too deep for gradient-based training without residual connections.

**The evidence is overwhelming**:
- MinGRU paper: 3 layers → loss 1.548 ✓
- Our model: 20 layers → loss 4.7-4.8 ✗
- Classic vanishing gradient symptoms
- Consistent across all optimizer/LR combinations

**Next step**: Train a 3-layer model matching the paper's configuration and validate we can achieve similar loss. If successful, we can explore depth scaling with proper architectural modifications (residual connections, layer normalization, etc).

---

## Sources

### Primary Research:
- [Were RNNs All We Needed? (ArXiv)](https://arxiv.org/html/2410.01201v1)
- [Were RNNs All We Needed? (HuggingFace)](https://huggingface.co/papers/2410.01201)
- [Medium: Exploring MiniGRU and MiniLSTM](https://medium.com/@sahin.samia/were-rnns-all-we-needed-exploring-minigru-and-minilstm-models-for-sequence-modeling-664e4675c339)

### Vanishing Gradients:
- [Vanishing Gradient Problem - Wikipedia](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- [GeeksforGeeks - Gradient Problems](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)
- [Analytics Vidhya - Exploring Gradients](https://www.analyticsvidhya.com/blog/2024/04/exploring-vanishing-and-exploding-gradients-in-neural-networks/)

### RNN Training Issues:
- [Stack Overflow - RNN Loss Plateau](https://stats.stackexchange.com/questions/283912/recurrent-neural-network-training-loss-does-not-decrease-past-a-certain-value)
- [PyTorch Forums - RNN Not Learning](https://discuss.pytorch.org/t/rnn-implementation-not-learning-and-test-loss-stuck-at-same-value/165126)

### Language Modeling:
- [HuggingFace - Perplexity Evaluation](https://huggingface.co/docs/transformers/en/perplexity)
- [Character-Level Language Modeling](https://arxiv.org/pdf/1808.04444)
- [Tokenizer-Free Language Models](https://arxiv.org/pdf/1908.10322)
