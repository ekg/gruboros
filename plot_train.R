#!/usr/bin/env Rscript
# Load required libraries
library(ggplot2)
library(minpack.lm)  # For better nonlinear fitting

# Parse command line arguments for table name and optional target loss
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 1) {
  stop("Usage: script.R <table_name> [target_loss]")
}
table_name <- args[1]
target_loss <- if(length(args) >= 2) as.numeric(args[2]) else NULL

# Read the data
data <- read.delim(table_name)

# Create a group column for plotting
data$group <- ifelse(is.na(data$val_loss), "train", "validation")

# Filter data to get only training loss - do this AFTER creating the group column
train_data <- subset(data, group == "train")

# Create the plot excluding step == 0, with specific aesthetic mapping and limits
p <- ggplot(subset(data, step != 0), aes(x = step, y = train_loss, color = group)) +
  geom_line() +
  labs(color = "group")  # Change legend title

# Fit nonlinear models to training loss
# Exponential decay model: a * exp(-b * x) + c
exp_model <- tryCatch({
  nlsLM(train_loss ~ a * exp(-b * step) + c, 
       data = train_data,
       start = list(a = 2, b = 0.001, c = 1),
       control = nls.lm.control(maxiter = 100))
}, error = function(e) NULL)

# Power law model: a * (step + d)^(-b) + c  
power_model <- tryCatch({
  nlsLM(train_loss ~ a * (step + d)^(-b) + c, 
       data = train_data,
       start = list(a = 5, b = 0.3, c = 1, d = 10),
       control = nls.lm.control(maxiter = 100))
}, error = function(e) NULL)

# Check if models were successfully fitted
models <- list()
if(!is.null(exp_model)) models$exp <- exp_model
if(!is.null(power_model)) models$power <- power_model

if(length(models) > 0) {
  # Compare models using AIC
  aic_values <- sapply(models, AIC)
  best_model <- models[[which.min(aic_values)]]
  best_model_name <- names(which.min(aic_values))
  
  # Get model parameters
  params <- coef(best_model)
  
  # Create prediction function based on best model
  if(best_model_name == "exp") {
    pred_fun <- function(x) params["a"] * exp(-params["b"] * x) + params["c"]
    min_loss <- params["c"]  # Asymptotic minimum
  } else {
    pred_fun <- function(x) params["a"] * (x + params["d"])^(-params["b"]) + params["c"]
    min_loss <- params["c"]  # Asymptotic minimum
  }
  
  # Generate predictions for plotting
  prediction_steps <- seq(0, max(train_data$step) * 2, length.out = 1000)
  predictions <- data.frame(
    step = prediction_steps,
    train_loss = pred_fun(prediction_steps)
  )
  
  # Calculate where loss will be within 1% of its asymptotic minimum
  threshold <- min_loss * 1.01
  below_threshold <- prediction_steps[predictions$train_loss <= threshold]
  
  if(length(below_threshold) > 0) {
    steps_to_threshold <- min(below_threshold)
  } else {
    # If no values below threshold, estimate when it might happen
    if(best_model_name == "exp") {
      # For exponential decay, solve a*exp(-b*x) + c = threshold
      # x = -ln((threshold-c)/a)/b
      steps_to_threshold <- -log((threshold - params["c"]) / params["a"]) / params["b"]
    } else {
      # For power law, solve a*(x+d)^(-b) + c = threshold
      # x = ((threshold-c)/a)^(-1/b) - d
      steps_to_threshold <- ((threshold - params["c"]) / params["a"])^(-1/params["b"]) - params["d"]
    }
    
    # Check if the solution is valid
    if(is.nan(steps_to_threshold) || is.infinite(steps_to_threshold) || steps_to_threshold < 0) {
      steps_to_threshold <- NA
    }
  }
  
  # Calculate steps to reach target loss if provided
  steps_to_target <- NA
  if(!is.null(target_loss)) {
    # Check if target is achievable (greater than asymptotic minimum)
    if(target_loss > min_loss) {
      below_target <- prediction_steps[predictions$train_loss <= target_loss]
      
      if(length(below_target) > 0) {
        steps_to_target <- min(below_target)
      } else {
        # If no values below target in our prediction range, estimate when it might happen
        if(best_model_name == "exp") {
          # For exponential decay, solve a*exp(-b*x) + c = target_loss
          # x = -ln((target_loss-c)/a)/b
          steps_to_target <- -log((target_loss - params["c"]) / params["a"]) / params["b"]
        } else {
          # For power law, solve a*(x+d)^(-b) + c = target_loss
          # x = ((target_loss-c)/a)^(-1/b) - d
          steps_to_target <- ((target_loss - params["c"]) / params["a"])^(-1/params["b"]) - params["d"]
        }
        
        # Check if the solution is valid
        if(is.nan(steps_to_target) || is.infinite(steps_to_target) || steps_to_target < 0) {
          steps_to_target <- NA
        }
      }
    } else {
      cat(paste("Warning: Target loss", target_loss, "is below the estimated asymptotic minimum", 
                min_loss, "and may never be reached.\n"))
    }
  }
  
  # Calculate model statistics
  model_summary <- summary(best_model)
  r_squared <- 1 - sum(residuals(best_model)^2) / sum((train_data$train_loss - mean(train_data$train_loss))^2)
  
  # Print model analysis
  cat("\nModel Analysis:\n")
  cat(paste("Best fitting model:", best_model_name, "decay\n"))
  cat(paste("Model parameters:", paste(names(params), "=", round(params, 5), collapse=", "), "\n"))
  cat(paste("R-squared:", round(r_squared, 4), "\n"))
  cat(paste("Estimated asymptotic minimum loss:", round(min_loss, 5), "\n"))
  cat(paste("Steps to reach within 1% of minimum:", round(steps_to_threshold, 0), "\n"))
  if(!is.null(target_loss)) {
    if(!is.na(steps_to_target)) {
      cat(paste("Steps to reach target loss", target_loss, ":", round(steps_to_target, 0), "\n\n"))
    } else {
      cat(paste("Unable to estimate steps to reach target loss", target_loss, "\n\n"))
    }
  } else {
    cat("\n")
  }
  
  # Format parameters for display
  param_text <- paste(names(params), "=", format(round(params, 4), nsmall=4), collapse="\n")
  
  # Create equation text based on model type
  if(best_model_name == "exp") {
    equation <- "y = a * exp(-b * x) + c"
  } else {
    equation <- "y = a * (x + d)^(-b) + c"
  }
  
  # Add the fitted curve to the plot
  p <- p + 
    geom_line(data = predictions, aes(x = step, y = train_loss), 
              color = "darkgreen", size = 1, linetype = "dashed") +
    annotate("text", x = max(train_data$step) * 0.7, y = max(train_data$train_loss) * 0.8,
             label = paste0(best_model_name, " model fit\n",
                            "Equation: ", equation, "\n",
                            "Min loss ≈ ", round(min_loss, 3), 
                            "\nSteps to min ≈ ", 
                            ifelse(is.na(steps_to_threshold), 
                                   "not reached", 
                                   format(round(steps_to_threshold, 0), big.mark=",")),
                            if(!is.null(target_loss) && !is.na(steps_to_target)) 
                              paste0("\nSteps to ", target_loss, " ≈ ", 
                                     format(round(steps_to_target, 0), big.mark=","))
                            else "",
                            "\n\nParameters:\n", param_text),
             hjust = 0, size = 3.5)
} else {
  cat("\nWarning: Failed to fit decay models. Check your data or try different starting parameters.\n\n")
}

# Save the plot as a PDF (10 x 5 inches)
output_file <- paste0(table_name, ".pdf")
ggsave(filename = output_file, plot = p, width = 10, height = 5, units = "in")
