# Introduction
options(stringsAsFactors = FALSE)
set.seed(123)

# Packages
library(tidyverse)
library(caret)
library(MASS)
library(glmnet)
library(pROC)
library(broom)
library(ggplot2)
library(forcats)
library(data.table)

# Load data
data_path <- "ikea_rcmp_simulated.csv"
df <- read.csv(data_path)

# Inspect
cat("Rows:", nrow(df), "  Cols:", ncol(df), "\n")
print(head(df))
print(str(df))

# Basic cleaning & types
df <- df %>%
  mutate(
    channel_type = factor(channel_type, levels = c("In-store","Remote/Digital","Hybrid")),
    channel = as.factor(channel),
    country = as.factor(country),
    contact_reason = as.factor(contact_reason),
    product_category = as.factor(product_category),
    language = as.factor(language),
    agent_id = as.factor(agent_id),
    is_returning_customer = as.factor(is_returning_customer),
    promo_used = as.factor(promo_used),
    sla_breached = as.factor(sla_breached),
    weekday = as.integer(weekday),
    hour = as.integer(hour),
    wait_log = log1p(wait_time_sec),
    handle_log = log1p(handle_time_min)
  )

num_cols <- c("wait_time_sec","handle_time_min","queue_length","backlog_index",
              "order_value","agent_tenure_months","pre_interaction_sentiment",
              "weekday","hour","wait_log","handle_log","csat","fcr")
nums <- df[, intersect(num_cols, names(df))]

# Train/Test split
set.seed(42)
in_train <- createDataPartition(df$csat, p = 0.75, list = FALSE)
training <- df[in_train, ]
testing  <- df[-in_train, ]

cat("Train:", nrow(training), " Test:", nrow(testing), "\n")

# EDA
ggplot(training, aes(x = csat)) +
  geom_histogram(binwidth = 1) +
  labs(title = "CSAT distribution", x = "CSAT", y = "Count") +
  theme_minimal()

ggplot(training, aes(x = wait_time_sec, y = csat, color = channel_type)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "CSAT vs Wait Time by Channel Type") +
  theme_minimal()

# Baseline CSAT
csat_mean <- mean(training$csat)
pred_base_csat <- rep(csat_mean, nrow(testing))
base_csat_metrics <- postResample(pred = pred_base_csat, obs = testing$csat)
cat("\n[Baseline CSAT] RMSE:", base_csat_metrics["RMSE"], "  R2:", base_csat_metrics["Rsquared"], "\n")

# Baseline FCR
fcr_majority <- as.integer(mean(as.integer(as.character(training$fcr))) >= 0.5)
pred_base_fcr <- rep(fcr_majority, nrow(testing))
base_acc <- mean(pred_base_fcr == as.integer(as.character(testing$fcr)))
cat("[Baseline FCR] Accuracy:", round(base_acc,4), "\n")

# CSAT Linear Model
csat_formula <- csat ~ channel_type * wait_log +
  channel + handle_log + queue_length + backlog_index + sla_breached +
  country + is_returning_customer + order_value + promo_used +
  weekday + hour + agent_tenure_months + pre_interaction_sentiment

csat_lm <- lm(csat_formula, data = training)
cat("\n[CSAT Linear Model]\n")
print(summary(csat_lm))

pred_csat_lm <- predict(csat_lm, newdata = testing)
csat_lm_metrics <- postResample(pred = pred_csat_lm, obs = testing$csat)
cat("[CSAT LM] RMSE:", csat_lm_metrics["RMSE"], " R2:", csat_lm_metrics["Rsquared"], "\n")

# Stepwise AIC selection
csat_step <- stepAIC(csat_lm, direction = "both", trace = FALSE)
cat("\n[CSAT Stepwise AIC]\n")
print(summary(csat_step))
pred_csat_step <- predict(csat_step, newdata = testing)
csat_step_metrics <- postResample(pred = pred_csat_step, obs = testing$csat)
cat("[CSAT Stepwise] RMSE:", csat_step_metrics["RMSE"], " R2:", csat_step_metrics["Rsquared"], "\n")

# CSAT glmnet
dv <- dummyVars(~ channel_type + channel + sla_breached + country +
                  is_returning_customer + promo_used + contact_reason + product_category + language,
                data = training, fullRank = TRUE)

X_train_cat <- predict(dv, newdata = training)
X_test_cat  <- predict(dv, newdata = testing)

X_train <- cbind(
  X_train_cat,
  dplyr::select(training, wait_time_sec, handle_time_min, queue_length, backlog_index, order_value,
                agent_tenure_months, pre_interaction_sentiment, weekday, hour, wait_log, handle_log)
) %>% as.matrix()

X_test <- cbind(
  X_test_cat,
  dplyr::select(testing, wait_time_sec, handle_time_min, queue_length, backlog_index, order_value,
                agent_tenure_months, pre_interaction_sentiment, weekday, hour, wait_log, handle_log)
) %>% as.matrix()

y_train_csat <- training$csat

set.seed(123)
csat_glmnet <- cv.glmnet(X_train, y_train_csat, alpha = 0.5, family = "gaussian", nfolds = 5)
cat("\n[CSAT glmnet] lambda.min:", csat_glmnet$lambda.min, "  lambda.1se:", csat_glmnet$lambda.1se, "\n")
pred_csat_glmnet <- predict(csat_glmnet, s = "lambda.min", newx = X_test)
csat_glmnet_metrics <- postResample(pred = as.numeric(pred_csat_glmnet), obs = testing$csat)
cat("[CSAT glmnet] RMSE:", csat_glmnet_metrics["RMSE"], " R2:", csat_glmnet_metrics["Rsquared"], "\n")

# FCR Logistic Model
fcr_formula <- fcr ~ channel_type + channel + wait_log + handle_log +
  queue_length + backlog_index + sla_breached + country +
  is_returning_customer + order_value + promo_used +
  weekday + hour + agent_tenure_months + pre_interaction_sentiment

fcr_glm <- glm(fcr_formula, data = training, family = binomial())
cat("\n[FCR Logistic Model]\n")
print(summary(fcr_glm))

pred_fcr_prob <- predict(fcr_glm, newdata = testing, type = "response")
pred_fcr_cls  <- ifelse(pred_fcr_prob > 0.5, 1, 0)
acc_fcr_glm <- mean(pred_fcr_cls == testing$fcr)
auc_fcr_glm <- pROC::auc(response = testing$fcr, predictor = pred_fcr_prob)
cat("[FCR Logistic] Accuracy:", round(acc_fcr_glm,4), " AUC:", round(as.numeric(auc_fcr_glm),4), "\n")

# FCR glmnet
y_train_fcr <- training$fcr
set.seed(123)
fcr_glmnet <- cv.glmnet(X_train, y_train_fcr, alpha = 0.5, family = "binomial", nfolds = 5)
cat("\n[FCR glmnet] lambda.min:", fcr_glmnet$lambda.min, "  lambda.1se:", fcr_glmnet$lambda.1se, "\n")

pred_fcr_prob_glmnet <- predict(fcr_glmnet, s = "lambda.min", newx = X_test, type = "response")
pred_fcr_cls_glmnet  <- ifelse(pred_fcr_prob_glmnet > 0.5, 1, 0)
acc_fcr_glmnet <- mean(pred_fcr_cls_glmnet == testing$fcr)
auc_fcr_glmnet <- pROC::auc(response = testing$fcr, predictor = as.numeric(pred_fcr_prob_glmnet))
cat("[FCR glmnet] Accuracy:", round(acc_fcr_glmnet,4), " AUC:", round(as.numeric(auc_fcr_glmnet),4), "\n")

# Effect sizes
cat("\n[Top drivers for CSAT (LM coefficients)]\n")
print(tidy(csat_step) %>% arrange(desc(abs(estimate))) %>% head(20))

cat("\n[Non-zero coefficients for FCR (glmnet)]\n")
coef_mat <- as.matrix(coef(fcr_glmnet, s = "lambda.min"))
nz <- coef_mat[coef_mat[,1]!=0, , drop = FALSE]
print(head(nz[order(abs(nz[,1]), decreasing = TRUE), , drop = FALSE], 25))

# Visual checks
pred_df <- data.frame(obs = testing$csat, pred = as.numeric(pred_csat_glmnet))
ggplot(pred_df, aes(x = pred, y = obs)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  labs(title = "CSAT Observed vs Predicted (glmnet)", x = "Predicted", y = "Observed") +
  theme_bw()

roc_obj <- roc(response = testing$fcr, predictor = as.numeric(pred_fcr_prob_glmnet))
plot(roc_obj, main = paste0("FCR ROC (AUC = ", round(as.numeric(auc_fcr_glmnet), 3), ")"))

# Save outputs
saveRDS(list(
  csat_lm = csat_lm,
  csat_step = csat_step,
  csat_glmnet = csat_glmnet,
  fcr_glm = fcr_glm,
  fcr_glmnet = fcr_glmnet,
  dv = dv
), file = "ikea_models.rds")

write.csv(pred_df, "ikea_csat_predictions.csv", row.names = FALSE)
