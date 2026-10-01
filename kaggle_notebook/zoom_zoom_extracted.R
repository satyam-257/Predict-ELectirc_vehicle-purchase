# --- Cell 0 (code) ---
options(stringsAsFactors = FALSE)
N_INNER <- 3L
SEED <- 51L
N_THREADS <- 4L
TARGET <- "Will_Buy_EV"
REFERENCE_PATH <- "/kaggle/input/notebooks/jazivxt/single-model-zoom-zoom/submission.csv"
if (!requireNamespace("lightgbm", quietly = TRUE)) install.packages("lightgbm", repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(lightgbm)
  library(dplyr)
  library(tibble)
  library(readr)
  library(ggplot2)
  library(scales)
  library(jsonlite)
})
input_files <- list.files("/kaggle/input", pattern = "^(train|test)\\.csv$", recursive = TRUE, full.names = TRUE)
train_path <- input_files[grepl("playground-series-s6e9/train\\.csv$", input_files)][1L]
test_path <- input_files[grepl("playground-series-s6e9/test\\.csv$", input_files)][1L]
stopifnot(!is.na(train_path), !is.na(test_path), file.exists(train_path), file.exists(test_path))
train <- read_csv(train_path, show_col_types = FALSE)
test <- read_csv(test_path, show_col_types = FALSE)
reference <- read_csv(REFERENCE_PATH, show_col_types = FALSE)
y <- as.integer(train[[TARGET]] %in% c("Yes", 1, TRUE))
predictors <- setdiff(names(train), c("id", TARGET))
train_raw <- as.data.frame(train[, predictors])
test_raw <- as.data.frame(test[, predictors])
stopifnot(ncol(train_raw) == 13L, identical(names(train_raw), names(test_raw)))

make_stratified_folds <- function(y, k, seed) {
  set.seed(seed)
  fold_id <- integer(length(y))
  for (cls in sort(unique(y))) {
    rows <- sample(which(y == cls))
    fold_id[rows] <- rep(seq_len(k), length.out = length(rows))
  }
  split(seq_along(y), fold_id)
}

frequency_map <- function(fit_values, new_values, normalize = FALSE) {
  lev <- unique(fit_values)
  codes <- match(fit_values, lev)
  counts <- tabulate(codes, nbins = length(lev))
  if (normalize) counts <- counts / sum(counts)
  new_codes <- match(new_values, lev)
  out <- numeric(length(new_values))
  known <- !is.na(new_codes)
  out[known] <- counts[new_codes[known]]
  out
}

weighted_tabulate <- function(index, weights, n) {
  totals <- numeric(n)
  grouped <- rowsum(weights, group = index, reorder = FALSE)
  totals[as.integer(rownames(grouped))] <- as.numeric(grouped[, 1L])
  totals
}

target_map <- function(fit_values, fit_y, new_values, smooth = 10) {
  lev <- unique(fit_values)
  codes <- match(fit_values, lev)
  counts <- tabulate(codes, nbins = length(lev))
  sums <- weighted_tabulate(codes, fit_y, length(lev))
  rates <- (sums + smooth * mean(fit_y)) / (counts + smooth)
  new_codes <- match(new_values, lev)
  out <- rep(mean(fit_y), length(new_values))
  known <- !is.na(new_codes)
  out[known] <- rates[new_codes[known]]
  out
}

auc_score <- function(y_true, p) {
  r <- rank(p, ties.method = "average")
  n1 <- sum(y_true == 1L)
  n0 <- sum(y_true == 0L)
  (sum(r[y_true == 1L]) - as.numeric(n1) * (n1 + 1) / 2) /
    (as.numeric(n1) * as.numeric(n0))
}

map_categories <- function(frame) {
  maps <- list(
    Gender = c(Female = 0, Male = 1, Other = 2),
    City_Type = c(Rural = 0, Suburban = 1, Urban = 2),
    Current_Car_Type = c(Hatchback = 0, Sedan = 1, SUV = 2, Truck = 3),
    Home_Charging_Possible = c(No = 0, Yes = 1),
    Subsidy_Available = c(No = 0, Yes = 1),
    Range_Anxiety_Level = c(High = 0, Low = 1, Medium = 2)
  )
  out <- frame
  for (nm in names(maps)) out[[nm]] <- unname(maps[[nm]][as.character(frame[[nm]])])
  out
}

neighbor_qte <- function(fit_income, fit_y, new_income, inner_folds, q = 1280L) {
  edges <- unique(as.numeric(quantile(fit_income, seq(0, 1, length.out = q + 1L), na.rm = TRUE)))
  interior <- edges[-c(1L, length(edges))]
  codes <- findInterval(fit_income, interior, left.open = TRUE) + 1L
  new_codes <- findInterval(new_income, interior, left.open = TRUE) + 1L
  n_bins <- length(edges)
  position <- (codes - 1L) / max(q - 1L, 1L)
  new_position <- (new_codes - 1L) / max(q - 1L, 1L)
  central_oof <- neighbor_oof <- rep(mean(fit_y), length(fit_y))
  kernel <- exp(-0.5 * (c(-1, 0, 1) / 0.8)^2)
  for (held in inner_folds) {
    fit <- setdiff(seq_along(fit_y), held)
    sums <- weighted_tabulate(codes[fit], fit_y[fit], n_bins)
    counts <- tabulate(codes[fit], nbins = n_bins)
    prior <- mean(fit_y[fit])
    center <- (sums + 10 * prior) / (counts + 10)
    ns <- kernel[1] * c(0, sums[-n_bins]) + kernel[2] * sums + kernel[3] * c(sums[-1], 0)
    nc <- kernel[1] * c(0, counts[-n_bins]) + kernel[2] * counts + kernel[3] * c(counts[-1], 0)
    neighbor <- (ns + 10 * sum(kernel) * prior) / (nc + 10 * sum(kernel))
    central_oof[held] <- center[codes[held]]
    neighbor_oof[held] <- neighbor[codes[held]]
  }
  sums <- weighted_tabulate(codes, fit_y, n_bins)
  counts <- tabulate(codes, nbins = n_bins)
  center <- (sums + 10 * mean(fit_y)) / (counts + 10)
  ns <- kernel[1] * c(0, sums[-n_bins]) + kernel[2] * sums + kernel[3] * c(sums[-1], 0)
  nc <- kernel[1] * c(0, counts[-n_bins]) + kernel[2] * counts + kernel[3] * c(counts[-1], 0)
  neighbor <- (ns + 10 * sum(kernel) * mean(fit_y)) / (nc + 10 * sum(kernel))
  list(train = cbind(position, central_oof, neighbor_oof),
       valid = cbind(new_position, center[new_codes], neighbor[new_codes]))
}

strong_local_features <- function(train_frame, train_y, valid_frame, inner_k = 3L, seed = 42L) {
  raw_train <- map_categories(train_frame)
  raw_valid <- map_categories(valid_frame)
  x_train <- as.data.frame(raw_train)
  x_valid <- as.data.frame(raw_valid)
  for (col in names(raw_train)) {
    a <- as.numeric(raw_train[[col]])
    b <- as.numeric(raw_valid[[col]])
    for (power in -4:3) {
      scale <- 10^power
      nm <- paste0(col, "_digit", power)
      x_train[[nm]] <- floor(a / scale) %% 10
      x_valid[[nm]] <- floor(b / scale) %% 10
    }
    ka <- as.character(train_frame[[col]])
    kb <- as.character(valid_frame[[col]])
    x_train[[paste0(col, "_freq")]] <- frequency_map(ka, ka, normalize = TRUE)
    x_valid[[paste0(col, "_freq")]] <- frequency_map(ka, kb, normalize = TRUE)
  }
  inner <- make_stratified_folds(train_y, inner_k, seed)
  for (col in names(train_frame)) {
    ka <- as.character(train_frame[[col]])
    kb <- as.character(valid_frame[[col]])
    encoded <- rep(NA_real_, length(train_y))
    for (held in inner) {
      fit <- setdiff(seq_along(train_y), held)
      encoded[held] <- target_map(ka[fit], train_y[fit], ka[held])
    }
    x_train[[paste0(col, "_te")]] <- encoded
    x_valid[[paste0(col, "_te")]] <- target_map(ka, train_y, kb)
  }
  income <- neighbor_qte(train_frame$Annual_Income_USD, train_y,
                         valid_frame$Annual_Income_USD, inner, q = 1280L)
  colnames(income$train) <- colnames(income$valid) <-
    c("income_q1280_pos", "income_q1280_center", "income_q1280_neighbor")
  x_train <- cbind(x_train, as.data.frame(income$train))
  x_valid <- cbind(x_valid, as.data.frame(income$valid))
  stopifnot(ncol(x_train) == 146L, ncol(x_valid) == 146L)
  list(train = x_train, valid = x_valid)
}


N_THREADS <- max(1L, parallel::detectCores(logical = TRUE))
folds <- make_stratified_folds(y, 5L, SEED)
valid_idx <- folds[[1L]]
fit_idx <- setdiff(seq_along(y), valid_idx)
message(sprintf("Seed %d holdout | %d fit / %d validation | %d threads", SEED, length(fit_idx), length(valid_idx), N_THREADS))

lgb_params <- list(
  objective = "binary", metric = "auc", learning_rate = 0.02,
  max_depth = 5L, num_leaves = 247L, min_data_in_leaf = 10L,
  feature_fraction = 0.3029300829885024,
  bagging_fraction = 0.812763123433567, bagging_freq = 1L,
  lambda_l1 = 0.07094285437903122, lambda_l2 = 2.0330390977032425,
  max_bin = 1024L, feature_pre_filter = FALSE,
  num_threads = N_THREADS, verbosity = -1L, seed = 60L
)

holdout_raw <- rbind(train_raw[valid_idx, , drop = FALSE], test_raw)
holdout_features <- strong_local_features(
  train_raw[fit_idx, , drop = FALSE], y[fit_idx], holdout_raw,
  N_INNER, SEED + 1L
)
x_fit <- as.matrix(holdout_features$train)
x_holdout <- as.matrix(holdout_features$valid)
n_valid <- length(valid_idx)
x_valid <- x_holdout[seq_len(n_valid), , drop = FALSE]
x_test_holdout <- x_holdout[n_valid + seq_len(nrow(test)), , drop = FALSE]
stopifnot(ncol(x_fit) == 146L, ncol(x_valid) == 146L,
          identical(colnames(x_fit), colnames(x_valid)))
d_fit <- lgb.Dataset(x_fit, label = y[fit_idx])
d_valid <- lgb.Dataset(x_valid, label = y[valid_idx], reference = d_fit)
validation_model <- lgb.train(
  params = lgb_params, data = d_fit, nrounds = 20000L,
  valids = list(valid = d_valid), early_stopping_rounds = 500L,
  verbose = -1L
)
validation_pred <- as.numeric(predict(
  validation_model, x_valid, num_iteration = validation_model$best_iter
))
validation_auc <- auc_score(y[valid_idx], validation_pred)
best_iteration <- validation_model$best_iter
message(sprintf("Seed %d validation AUC %.9f | best iteration %d",
                SEED, validation_auc, best_iteration))

rm(holdout_features, holdout_raw, x_fit, x_holdout, x_valid,
   x_test_holdout, d_fit, d_valid, validation_model)
invisible(gc(FALSE))

full_features <- strong_local_features(train_raw, y, test_raw, N_INNER, SEED)
x_train <- as.matrix(full_features$train)
x_test <- as.matrix(full_features$valid)
stopifnot(ncol(x_train) == 146L, ncol(x_test) == 146L,
          identical(colnames(x_train), colnames(x_test)))
d_train <- lgb.Dataset(x_train, label = y)
final_model <- lgb.train(
  params = lgb_params, data = d_train,
  nrounds = best_iteration, verbose = -1L
)
pred <- as.numeric(predict(final_model, x_test, num_iteration = best_iteration))
stopifnot(length(pred) == nrow(test), all(is.finite(pred)),
          all(pred >= 0 & pred <= 1), !anyDuplicated(test$id))

stopifnot(all(c("id", TARGET) %in% names(reference)),
          !anyDuplicated(reference$id), nrow(reference) == nrow(test))
ref_idx <- match(test$id, reference$id)
stopifnot(!anyNA(ref_idx),
          identical(as.character(reference$id[ref_idx]), as.character(test$id)))
reference_pred <- pmin(1, pmax(0, as.numeric(reference[[TARGET]][ref_idx])))
stopifnot(all(is.finite(reference_pred)))
rank01 <- function(x) (rank(x, ties.method = "average") - 0.5) / length(x)
reference_rank <- rank01(reference_pred)
model_rank <- rank01(pred)
blend_pred <- 0.80 * reference_rank + 0.20 * model_rank

submission <- tibble(id = test$id, Will_Buy_EV = pred)
submissionblend <- tibble(id = test$id, Will_Buy_EV = blend_pred)
stopifnot(identical(submission$id, test$id),
          identical(submissionblend$id, test$id),
          all(is.finite(blend_pred)), all(blend_pred >= 0 & blend_pred <= 1))
write_csv(submission, "submission.csv")
write_csv(submissionblend, "submissionblend.csv")
write_csv(tibble(seed = SEED, features = ncol(x_train), learning_rate = 0.02,
  max_depth = 5L, num_leaves = 247L, threads = N_THREADS,
  validation_auc = validation_auc, best_iteration = best_iteration), "cv_summary.csv")

roc_order <- order(validation_pred, decreasing = TRUE)
roc_y <- y[valid_idx][roc_order]
roc_df <- tibble(
  false_positive_rate = c(0, cumsum(1 - roc_y) / sum(1 - roc_y)),
  true_positive_rate = c(0, cumsum(roc_y) / sum(roc_y))
)
roc_plot <- ggplot(roc_df, aes(false_positive_rate, true_positive_rate)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey65") +
  geom_path(color = "#0072B2", linewidth = 1) +
  coord_equal() +
  labs(title = sprintf("Seed 51 ROC · AUC %.6f", validation_auc),
       x = "False positive rate", y = "True positive rate") +
  theme_minimal(base_size = 12)
importance_df <- as_tibble(lgb.importance(final_model)) |>
  slice_max(Gain, n = 24) |>
  arrange(Gain)
importance_plot <- ggplot(importance_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#009E73", width = 0.72) + coord_flip() +
  labs(title = "LightGBM feature importance", x = NULL, y = "Gain") +
  theme_minimal(base_size = 12)
prediction_df <- tibble(
  reference_rank = reference_rank,
  model_rank = model_rank,
  blend_rank = blend_pred
) |>
  tidyr::pivot_longer(everything(), names_to = "prediction", values_to = "score")
prediction_plot <- ggplot(prediction_df, aes(score, color = prediction, fill = prediction)) +
  geom_density(alpha = 0.16, linewidth = 0.75) +
  scale_color_manual(values = c(reference_rank = "#0072B2", model_rank = "#009E73",
                                blend_rank = "#CC79A7")) +
  scale_fill_manual(values = c(reference_rank = "#0072B2", model_rank = "#009E73",
                               blend_rank = "#CC79A7")) +
  labs(title = "Ranked prediction distributions", x = "Rank score", y = "Density",
       color = NULL, fill = NULL) +
  theme_minimal(base_size = 12)
print(roc_plot)
print(importance_plot)
print(prediction_plot)
ggsave("validation_roc.png", roc_plot, width = 6, height = 5, dpi = 160)
ggsave("lightgbm_importance.png", importance_plot, width = 8, height = 7, dpi = 160)
ggsave("prediction_distributions.png", prediction_plot, width = 8, height = 4.5, dpi = 160)
message("submission.csv and submissionblend.csv saved")



# --- Cell 1 (code) ---
cat("Ｈ𝐀𝑷𝑷𝓎 🇰𝗮𝘨𝗴🇱𝖎Ｎɢ 💯\n")

