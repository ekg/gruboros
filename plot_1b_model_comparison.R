#!/usr/bin/env Rscript
# 1B Model Training Comparison: HasteGRU vs Mamba2
# CORRECTED PADDING: Only runs with actual_length loss masking (Dec 22+ commits)
# PyTorch DDP, 8 GPUs, batch=24, chunk=512

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

# === CORRECTED PADDING RUNS ONLY ===
# These runs have actual_length loss masking (commit 80962f8, Dec 22 2025)
# All use: batch=24, chunk=512, 8 GPUs = 98,304 tokens/step
models <- list(
  # HasteGRU_MultLM: Fused CUDA kernel, 1.01B params, depth=27
  list(log = "logs/haste_mult_gru_1b_20251222_182059.log", name = "HasteGRU_Mult", params = "1.01B"),

  # Mamba2LM: SSD kernel, 1.00B params, depth=35
  list(log = "logs/mamba2_1b_matched_20251223_072401.log", name = "Mamba2 SSD", params = "1.00B")
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

# Define color palette
n_models <- length(unique(all_data$model))
colors <- c(
  "#009E73",  # Green - HasteGRU_Mult
  "#0072B2"   # Blue - Mamba2 SSD
)[1:n_models]

# Create the plot with raw data as faint background + smoothed overlay
p <- ggplot(all_data, aes(x = step, color = model)) +
  # Raw data as faint lines
  geom_line(aes(y = loss), alpha = 0.15, linewidth = 0.3) +
  # Smoothed overlay
  geom_line(aes(y = loss_smooth), linewidth = 1.2, alpha = 0.9) +
  scale_y_log10(
    breaks = c(3, 4, 5, 6, 7, 8, 10, 12),
    labels = c("3", "4", "5", "6", "7", "8", "10", "12")
  ) +
  annotation_logticks(sides = "l") +
  scale_x_continuous(labels = scales::comma, limits = c(0, 3000)) +
  scale_color_manual(values = colors) +
  labs(
    title = "HasteGRU vs Mamba2: Corrected Padding Comparison",
    subtitle = "Loss masking enabled - honest loss on real tokens only (batch=24, chunk=512, ~1B params)",
    x = "Training Step",
    y = "Loss (log scale)",
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    panel.grid.minor = element_blank(),
    legend.text = element_text(size = 12)
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
ggsave("/tmp/1b_model_comparison.png", p, width = 12, height = 7, dpi = 150)
cat("\nPlot saved to /tmp/1b_model_comparison.png\n")
