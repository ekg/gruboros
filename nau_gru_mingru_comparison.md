# NAU-GRU vs minGRU: Systematic Comparison

## 1. Activation Functions

### minGRU
- Uses `log_g(x)` activation:
  ```python
  def log_g(x):
      return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
  ```
- For positive inputs: `log(relu(x) + 0.5)`
- For negative inputs: `-softplus(-x) = -log(1 + exp(-x))`

### NAU-GRU
- Uses identical activation (lines 41-51):
  ```python
  # For positive h_t: log(relu(h_t) + 0.5)
  h_t_pos = tl.maximum(h_t, 0.0)
  h_new_pos = tl.log(h_t_pos + 0.5)
  
  # For negative h_t: -softplus(-h_t) = -log(1 + exp(-h_t))
  h_new_neg = -tl.log(1.0 + tl.exp(-h_t))
  
  h_new = tl.where(h_t >= 0, h_new_pos, h_new_neg)
  ```

**Verdict: IDENTICAL activation functions**

## 2. Update Equations and Log-Space Handling

### minGRU
- Sequential update (seq_len == 1):
  ```python
  log_gate = torch.log(gate_sigmoid.clamp(min=1e-8))
  log_one_minus_gate = torch.log((1 - gate_sigmoid).clamp(min=1e-8))
  log_out = torch.logaddexp(
      log_one_minus_gate + prev_hidden,
      log_gate + log_hidden
  )
  ```
- Parallel scan uses associative scan with log coefficients

### NAU-GRU
- Sequential update (in kernel):
  ```python
  # Gate log probabilities
  log_g = -tl.log(1.0 + tl.exp(-g_t))        # log(sigmoid(g_t))
  log_one_minus_g = -tl.log(1.0 + tl.exp(g_t))  # log(1 - sigmoid(g_t))
  
  # Log-sum-exp
  term1 = log_one_minus_g + h_log  # log((1-g)*exp(h_log))
  term2 = log_g + h_new            # log(g*exp(h_new))
  
  # Stable log-sum-exp
  max_val = tl.maximum(term1, term2)
  h_log = max_val + tl.log(tl.exp(term1 - max_val) + tl.exp(term2 - max_val))
  ```

**Key Differences:**
1. NAU-GRU uses direct softplus formulation: `-log(1 + exp(-g))` instead of `log(sigmoid.clamp())`
2. NAU-GRU implements manual log-sum-exp with max subtraction for stability
3. minGRU uses torch.logaddexp (which internally does the same max subtraction)

**Verdict: Mathematically equivalent but different numerical implementations**

## 3. Initialization Schemes

### minGRU
- **Weights**: Standard PyTorch initialization (Xavier/Kaiming)
- **Hidden state**: Not explicitly set in the class (handled externally)
  
### NAU-GRU
- **Weights**: Custom initialization
  ```python
  std = 0.02 / math.sqrt(dim)
  torch.nn.init.normal_(self.to_hidden_and_gate.weight, mean=0.0, std=std)
  torch.nn.init.constant_(self.to_out.weight, 0.0)  # Zero-initialized output
  ```
- **Hidden state**: Initialized to -2.0 in log space (exp(-2) ≈ 0.135)
  ```python
  prev_hidden = torch.full((B, self.dim_inner), -2.0, device=device, dtype=dtype)
  ```

**Verdict: DIFFERENT initialization strategies**

## 4. Computational Flow

### minGRU
- Two branches: sequential (seq_len == 1) vs parallel (seq_len > 1)
- Parallel branch uses associative scan
- Hidden states always maintained in log space between steps

### NAU-GRU
- Always sequential processing in Triton kernel
- Processes one time step at a time in a loop
- Hidden states maintained in log space throughout

**Key Difference:** minGRU can process entire sequences in parallel using associative scan, while NAU-GRU always processes sequentially.

## 5. Additional Features in NAU-GRU

### Energy Barriers (unique to NAU-GRU)
```python
if use_barriers:
    # Soft barrier forces that grow exponentially near boundaries
    lower_force = 0.5 * tl.exp(barrier_min - h_log + 5.0)
    upper_force = 0.5 * tl.exp(h_log - barrier_max + 5.0)
    
    # Apply corrections - push away from barriers
    h_log = h_log + lower_force - upper_force
    
    # Hard clamp as final safety
    h_log = tl.minimum(tl.maximum(h_log, barrier_min), barrier_max)
```

This feature adds repulsive forces near boundaries to prevent numerical explosion, not present in minGRU.

## Summary of Key Differences

1. **Activation**: Identical mathematical formulation
2. **Update equations**: Mathematically equivalent, different numerical implementations
3. **Initialization**: Different strategies (NAU-GRU more conservative)
4. **Processing**: minGRU supports parallel scan, NAU-GRU is always sequential
5. **Stability features**: NAU-GRU has unique energy barriers for numerical stability
6. **Output initialization**: NAU-GRU zeros output weights for residual connection

The core mathematical operations are nearly identical, but NAU-GRU adds stability features and uses more conservative initialization while sacrificing parallel processing capability.