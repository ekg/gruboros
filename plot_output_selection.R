#!/usr/bin/env Rscript
#
# Output Selection in Recurrent Language Models
#
# Compares Stock GRU (no output selection) vs architectures with output selection:
# - Mamba2: C matrix provides input-dependent output selection
# - Mult GRU: Multiplicative gate h' = h * sigma(Wx*x + Wh*h) filters outputs
#
# Key finding: Output selection mechanism accounts for ~0.8 nats improvement.

library(ggplot2)
library(dplyr)
library(tidyr)

# Parse log file for Step N: L=X.XX pattern
parse_log <- function(filepath, label) {
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
    model = label
  )
}

# Define logs with labels and colors (all 512 context)
logs <- list(
  # No Output Selection (Baseline)
  list(file = "logs/test_standard_gru_no_tbptt_10k_20251215_215912.log",
       label = "Stock GRU (no output selection)", color = "#CC0000"),

  # With Output Selection
  list(file = "logs/mamba2_1b_20251207_060701.log",
       label = "Mamba2 (C matrix selection)", color = "#005a8d"),
  list(file = "logs/cudnn_mult_gru_512_20251212_022323.log",
       label = "Mult GRU sigmoid", color = "#007a59"),
  list(file = "logs/cudnn_mult_gru_silu_20251216_220912.log",
       label = "Mult GRU SiLU (Mamba2-style)", color = "#9400D3")
)

# Parse all logs
all_data <- do.call(rbind, lapply(logs, function(l) {
  parse_log(l$file, l$label)
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

# Create the plot
p <- ggplot(all_data, aes(x = step, y = loss, color = model)) +
  # Raw data with low alpha
  geom_line(alpha = 0.15, linewidth = 0.3) +
  # Heavy LOESS smoothing (span controls smoothness, higher = smoother)
  geom_smooth(method = "loess", span = 0.3, se = FALSE, linewidth = 1.5) +
  scale_color_manual(values = color_map, name = NULL) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, 10500), ylim = c(2.5, 8.5)) +
  labs(
    title = "Output Selection in Recurrent LMs: Stock GRU vs Selective Architectures",
    subtitle = "~1B params, The Pile, 512-token chunks, 8x A100",
    x = "Training Step",
    y = "Cross-Entropy Loss (nats)"
  ) +
  annotate("text", x = 100, y = 7.8, label = "Random init: 10.82",
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

# Calculate average loss over last 1000 steps for each model
final_losses <- all_data %>%
  group_by(model) %>%
  summarize(
    max_step = max(step),
    # Average over last 1000 steps (or all steps if < 1000)
    avg_loss = mean(loss[step > max(step) - 1000]),
    n_steps = sum(step > max(step) - 1000),
    pct = if (max(step) < 9999) sprintf(" (%d%%)", as.integer(max(step)/100)) else "",
    .groups = "drop"
  )

# Save plot
ggsave("/tmp/output_selection_comparison.png", p,
       width = 14, height = 8, dpi = 150, bg = "white")

cat("Saved: /tmp/output_selection_comparison.png\n")

# Print average losses over last 1k steps
cat("\nAverage loss (last 1k steps):\n")
for (i in 1:nrow(final_losses)) {
  cat(sprintf("  %s: %.2f%s (n=%d)\n",
              final_losses$model[i],
              final_losses$avg_loss[i],
              final_losses$pct[i],
              final_losses$n_steps[i]))
}
