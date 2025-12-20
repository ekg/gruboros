#!/usr/bin/env Rscript
# 1B Model Training Comparison: cuDNN GRU variants vs Mamba SSM
# PyTorch DDP, 8 GPUs

library(ggplot2)
library(dplyr)
library(tidyr)

# Function to extract data from log file
extract_log_data <- function(log_file, model_name) {
  if (!file.exists(log_file)) {
    cat(sprintf("Warning: %s not found\n", log_file))
    return(NULL)
  }

  lines <- readLines(log_file, warn = FALSE)
  step_lines <- grep("^Step\\s+[0-9]+:", lines, value = TRUE)

  if (length(step_lines) == 0) {
    cat(sprintf("Warning: No step lines in %s\n", log_file))
    return(NULL)
  }

  # Extract step and loss
  steps <- as.numeric(gsub("^Step\\s+([0-9]+):.*", "\\1", step_lines))
  losses <- as.numeric(gsub(".*L=([0-9.]+).*", "\\1", step_lines))

  data.frame(step = steps, loss = losses, model = model_name)
}

# Define models to compare (Mamba2 baseline onwards + ablation study)
models <- list(
  # === Baselines ===
  list(log = "logs/mamba2_1b_20251207_060701.log", name = "Mamba2 SSD", params = "1.1B"),
  list(log = "logs/mamba2_ffn3_1b_20251207_234937.log", name = "Mamba2+FFN3", params = "1.0B"),
  list(log = "logs/cudnn_mult_gru_1b_20251208_170520.log", name = "Mult GRU (full)", params = "1.0B"),

  # === Ablation Study: What makes Mult GRU work? ===
  # Option 1: Input-only gate (remove W_h from gate)
  list(log = "logs/input_only_gate_1b_20251209_035046.log", name = "Ablation 1: Input-only gate", params = "1.0B"),

  # Option 2: EMA + input gate (cheapest recurrence)
  list(log = "logs/ema_input_gate_1b_20251209_023906.log", name = "Ablation 2: EMA + input gate", params = "1.1B"),

  # Option 3: GLU-style gate (h-dependent only, no x)
  list(log = "logs/glu_gate_1b_20251209_160229.log", name = "Ablation 3: GLU gate", params = "1.0B")

  # === ElmanSilu (haste CUDA, harmonized with Mamba2, no TBPTT) ===
  # Uncomment after training completes and update log filename:
  # list(log = "logs/elman_silu_1b_XXXXXXXX_XXXXXX.log", name = "ElmanSilu (haste)", params = "1.0B")
)

# Extract data from all logs
all_data <- data.frame()
for (m in models) {
  data <- extract_log_data(m$log, sprintf("%s (%s)", m$name, m$params))
  if (!is.null(data) && nrow(data) > 10) {
    all_data <- rbind(all_data, data)
    cat(sprintf("Loaded %s: %d steps, final loss %.4f\n", m$name, max(data$step), tail(data$loss, 1)))
  }
}

if (nrow(all_data) == 0) {
  stop("No data loaded!")
}

# Simple rolling mean using base R (no zoo package needed)
rolling_mean <- function(x, k = 50) {
  n <- length(x)
  result <- rep(NA_real_, n)
  for (i in k:n) {
    result[i] <- mean(x[(i - k + 1):i])
  }
  result
}

# Smooth the data with rolling mean for cleaner visualization
all_data <- all_data %>%
  group_by(model) %>%
  arrange(step) %>%
  mutate(loss_smooth = rolling_mean(loss, k = 50)) %>%
  ungroup()

# Define color palette (6 models)
n_models <- length(unique(all_data$model))
colors <- c(
  "#0072B2",  # Blue - Mamba2 SSD
  "#56B4E9",  # Sky blue - Mamba2+FFN3
  "#009E73",  # Green - Mult GRU (full)
  "#E69F00",  # Orange - Ablation 1: Input-only gate
  "#CC79A7",  # Pink - Ablation 2: EMA + input gate
  "#D55E00"   # Vermillion - Ablation 3: GLU gate
)[1:n_models]

# Create the plot
p <- ggplot(all_data, aes(x = step, y = loss_smooth, color = model)) +
  geom_line(linewidth = 1, alpha = 0.9) +
  scale_y_continuous(limits = c(2.5, 7), breaks = seq(2, 8, 0.5)) +
  scale_x_continuous(labels = scales::comma) +
  scale_color_manual(values = colors) +
  labs(
    title = "Multiplicative Gating Ablation Study: What makes Mult GRU work?",
    subtitle = "Baselines (Mamba2, Mult GRU) vs ablations removing different gating components",
    x = "Training Step",
    y = "Loss",
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.direction = "vertical",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    panel.grid.minor = element_blank(),
    legend.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(ncol = 2))

# Add final loss annotation for each model
final_losses <- all_data %>%
  group_by(model) %>%
  filter(step == max(step)) %>%
  slice(1) %>%
  ungroup()

# Print summary
cat("\n=== Final Loss Summary ===\n")
for (i in 1:nrow(final_losses)) {
  cat(sprintf("%s: Step %d, Loss %.4f\n",
              final_losses$model[i],
              final_losses$step[i],
              final_losses$loss[i]))
}

# Save plot
ggsave("/tmp/cudnn_ema_comparison.png", p, width = 14, height = 8, dpi = 150)
cat("\nPlot saved to /tmp/cudnn_ema_comparison.png\n")
