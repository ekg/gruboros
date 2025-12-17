#!/usr/bin/env Rscript
#
# Output Selection in Recurrent Language Models
#
# Compares Stock GRU (no output selection) vs architectures with output selection:
# - Mamba2: C matrix provides input-dependent output selection
# - Mult GRU: Multiplicative gate h' = h * sigma(Wx*x + Wh*h) filters outputs
#
# Key finding: Output selection mechanism accounts for ~0.8 nats improvement.
#
# IMPORTANT: X-axis is total tokens trained, not steps!
# Different models have different batch sizes, so steps aren't comparable.

library(ggplot2)
library(dplyr)
library(tidyr)

# Parse log file for Step N: L=X.XX pattern
parse_log <- function(filepath, label, tokens_per_step) {
  if (!file.exists(filepath)) {
    return(NULL)
  }
  lines <- readLines(filepath, warn = FALSE)
  matches <- regmatches(lines, regexec("Step\\s+(\\d+):\\s+L=([0-9.]+)", lines))
  matches <- matches[sapply(matches, length) > 0]

  if (length(matches) == 0) return(NULL)

  data.frame(
    step = as.integer(sapply(matches, `[`, 2)),
    loss = as.numeric(sapply(matches, `[`, 3)),
    model = label,
    tokens_per_step = tokens_per_step
  ) %>%
    mutate(tokens = step * tokens_per_step)
}

# Define logs with labels, colors, and tokens_per_step
# tokens_per_step = batch_size * chunk_size * num_gpus
logs <- list(
  # Stock GRU: batch=32, chunk=512, gpus=8 → 131,072 tokens/step
  list(file = "logs/test_standard_gru_no_tbptt_10k_20251215_215912.log",
       label = "Stock GRU (no output selection)", color = "#CC0000",
       tokens_per_step = 32 * 512 * 8),  # 131,072

  # Mamba2: batch=16, chunk=512, gpus=8 → 65,536 tokens/step
  list(file = "logs/mamba2_1b_20251207_060701.log",
       label = "Mamba2 (C matrix selection)", color = "#005a8d",
       tokens_per_step = 16 * 512 * 8),  # 65,536

  # Mult GRU sigmoid: batch=24, chunk=512, gpus=8 → 98,304 tokens/step
  list(file = "logs/cudnn_mult_gru_512_20251212_022323.log",
       label = "Mult GRU sigmoid", color = "#007a59",
       tokens_per_step = 24 * 512 * 8),  # 98,304

  # Mult GRU SiLU: batch=24, chunk=512, gpus=8 → 98,304 tokens/step
  list(file = "logs/cudnn_mult_gru_silu_20251216_220912.log",
       label = "Mult GRU SiLU (Mamba2-style)", color = "#9400D3",
       tokens_per_step = 24 * 512 * 8)   # 98,304
)

# Parse all logs
all_data <- do.call(rbind, lapply(logs, function(l) {
  parse_log(l$file, l$label, l$tokens_per_step)
}))

if (is.null(all_data) || nrow(all_data) == 0) {
  stop("No data found in any log files!")
}

# Create color mapping
color_map <- setNames(
  sapply(logs, `[[`, "color"),
  sapply(logs, `[[`, "label")
)

# Order models by their order in the logs list (for legend)
model_order <- sapply(logs, `[[`, "label")
all_data$model <- factor(all_data$model, levels = model_order)

# Convert tokens to billions for cleaner axis
all_data$tokens_B <- all_data$tokens / 1e9

# Find max tokens across all models for x-axis limit
max_tokens_B <- max(all_data$tokens_B)

# Create the plot with tokens on x-axis
p <- ggplot(all_data, aes(x = tokens_B, y = loss, color = model)) +
  # Raw data with low alpha
  geom_line(alpha = 0.15, linewidth = 0.3) +
  # Heavy LOESS smoothing (span controls smoothness, higher = smoother)
  geom_smooth(method = "loess", span = 0.3, se = FALSE, linewidth = 1.5) +
  scale_color_manual(values = color_map, name = NULL) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, max_tokens_B * 1.05), ylim = c(2.5, 8.5)) +
  labs(
    title = "Output Selection in Recurrent LMs: Stock GRU vs Selective Architectures",
    subtitle = "~1B params, The Pile, 512-token context, 8x A100 (x-axis: total tokens trained)",
    x = "Tokens Trained (Billions)",
    y = "Cross-Entropy Loss (nats)"
  ) +
  annotate("text", x = 0.02, y = 7.8, label = "Random init: 10.82",
           hjust = 0, size = 3.5, color = "#666666") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "#555555"),
    legend.position = "right",
    legend.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#cccccc", linewidth = 0.3),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  guides(color = guide_legend(ncol = 1))

# Calculate statistics at similar token counts for fair comparison
final_stats <- all_data %>%
  group_by(model) %>%
  summarize(
    max_step = max(step),
    max_tokens_B = max(tokens_B),
    tokens_per_step = first(tokens_per_step),
    # Average over last ~100M tokens worth of steps
    last_100M_steps = ceiling(100e6 / first(tokens_per_step)),
    avg_loss = mean(loss[step > max(step) - last_100M_steps]),
    n_steps = sum(step > max(step) - last_100M_steps),
    .groups = "drop"
  )

# Save plot
ggsave("/tmp/output_selection_comparison.png", p,
       width = 14, height = 8, dpi = 150, bg = "white")

cat("Saved: /tmp/output_selection_comparison.png\n")

# Print token-based statistics
cat("\nTraining configuration:\n")
for (i in 1:nrow(final_stats)) {
  cat(sprintf("  %s: %d tokens/step, %.2fB total tokens\n",
              final_stats$model[i],
              final_stats$tokens_per_step[i],
              final_stats$max_tokens_B[i]))
}

cat("\nAverage loss (last ~100M tokens):\n")
for (i in 1:nrow(final_stats)) {
  cat(sprintf("  %s: %.2f (n=%d steps)\n",
              final_stats$model[i],
              final_stats$avg_loss[i],
              final_stats$n_steps[i]))
}

# Print note about token discrepancy
cat("\n⚠️  NOTE: Stock GRU trained on 2x more tokens than Mamba2!\n")
cat("   Stock GRU: 131K tokens/step × 10K steps = 1.31B tokens\n")
cat("   Mamba2:     65K tokens/step × 10K steps = 0.65B tokens\n")
cat("   For fair comparison, look at same x-position (tokens), not steps.\n")
