## Load required packages
library(magrittr)
library(purrr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(reshape2)
library(plotly)
library(matrixStats)
library(dbscan)
library(MASS)
library(h2o)
library(Rtsne)
library(fastknn)
library(Matrix)
library(xgboost)

## Load toy example
load("./data/concentric_circles.rda")
glimpse(concentric.circles)

## Plot toy example
g <- ggplot(concentric.circles, aes(x, y, shape = class, color = class)) +
   geom_point(alpha = 1, size = 1.5) + 
   scale_shape_manual(name = "Class", values = c(4, 3)) +
   scale_color_manual(name = "Class", values = c('#0C4B8E', '#BF382A')) +
   guides(shape = guide_legend(barwidth = 0.5, barheight = 7)) +
   coord_fixed() +
   labs(x = expression(x[1]), y = expression(x[2])) +
   theme_ipsum(axis_title_size = 12)
plot(g)

## Create training data
dt.train <- concentric.circles

## Create test data
n <- 200   
x <- rep(seq(-1, 1, length = n), times = n)
y <- rep(seq(-1, 1, length = n), each = n)
dt.test <- data_frame(x = x, y = y)

## Train GLM model
glm.model <- glm(data = dt.train, formula = class ~ x + y, family = "binomial")
yhat <- predict(glm.model, dt.test, type = "response")

## Plot decision boundary for test data
g <- data_frame(x1 = x, x2 = y, y = yhat, z = ifelse(y >= 0.5, 1, 0)) %>% 
   ggplot() + 
   geom_tile(aes_string("x1", "x2", fill = "y"), color = NA, size = 0, alpha = 0.8) +
   scale_fill_distiller(name = "Prob +", palette = "Spectral", limits = c(0.38, 0.62)) +
   geom_point(data = dt.train, aes_string("x", "y", shape = "class"), 
              alpha = 1, size = 1.5, color = "black") + 
   geom_contour(aes_string("x1", "x2", z = "z"), color = 'red', alpha = 0.6, 
                size = 0.5, bins = 1) +
   scale_shape_manual(name = "Class", values = c(4, 3)) +
   guides(fill = guide_colorbar(barwidth = 0.5, barheight = 7),
          shape = guide_legend(barwidth = 0.5, barheight = 7)) +
   coord_fixed() +
   labs(x = expression(x[1]), y = expression(x[2])) +
   theme_ipsum(axis_title_size = 12)
plot(g)

## Create a third feature
dt.train <- dt.train %>% 
   mutate(z = x^2 + y^2)
dt.test <- dt.test %>% 
   mutate(z = x^2 + y^2)

## Plot 3d space
p <- dt.train %>%  
   plot_ly(x = ~x, y = ~y, z = ~z, color = ~class, colors = c('#0C4B8E', '#BF382A'), 
           symbol = ~class, symbols = c("x", "cross")) %>%
   add_markers() %>%
   layout(scene = list(
      xaxis = list(title = 'X1'), yaxis = list(title = 'X2'), 
      zaxis = list(title = 'X3 = X1² + X2²')
   ))
p

## Plot 3d surface boundary
p <- plot_ly() %>% 
   add_trace(type = "surface", x = seq(-1, 1, length = n), y = seq(-1, 1, length = n), 
             z = matrix(0.5, ncol = n, nrow = n), colors = c('gray20', 'gray80'), 
             color = c(0, 0.5, 1), showlegend = FALSE, name = "Decision Boundary", 
             surfacecolor = matrix(seq(0, 1, length = 200), ncol = n, nrow = n), opacity = 0.8) %>% 
   add_trace(type = "scatter3d", mode = "markers", x = dt.train$x[dt.train$class==0], 
             y = dt.train$y[dt.train$class==0], z = dt.train$z[dt.train$class==0], 
             marker = list(color = "#0C4B8E", symbol = "x"), name = "0") %>%
   add_trace(type = "scatter3d", mode = "markers", x = dt.train$x[dt.train$class==1], 
             y = dt.train$y[dt.train$class==1], z = dt.train$z[dt.train$class==1], 
             marker = list(color = "#BF382A", symbol = "cross"), name = "1") %>% 
   layout(scene = list(
      xaxis = list(title = 'X1'), yaxis = list(title = 'X2'), 
      zaxis = list(title = 'X3 = X1² + X2²')
   ))
p

## Train GLM including the new feature
glm.model <- glm(data = dt.train, formula = class ~ x + y + z, family = "binomial")
yhat <- predict(glm.model, dt.test, type = "response")

## Plot decision binary for the new 3d test set
g <- data_frame(x1 = x, x2 = y, y = yhat, z = ifelse(y >= 0.5, 1, 0)) %>% 
   ggplot() + 
   geom_tile(aes_string("x1", "x2", fill = "y"), color = NA, size = 0, alpha = 0.8) +
   scale_fill_distiller(name = "Prob +", palette = "Spectral", limits = c(0, 1)) +
   geom_point(data = dt.train, aes_string("x", "y", shape = "class"), 
              alpha = 1, size = 1.5, color = "black") + 
   geom_contour(aes_string("x1", "x2", z = "z"), color = 'red', alpha = 0.6, 
                size = 0.5, bins = 1) +
   scale_shape_manual(name = "Class", values = c(4, 3)) +
   guides(fill = guide_colorbar(barwidth = 0.5, barheight = 7),
          shape = guide_legend(barwidth = 0.5, barheight = 7)) +
   coord_fixed() +
   labs(x = expression(x[1]), y = expression(x[2])) +
   theme_ipsum(axis_title_size = 12)
plot(g)

## Load census dataset
load("./data/census_income.rda")
glimpse(census.income)

## Input missing values
fillMissingValue <- function(x) {
   if(is.numeric(x)) {
      fill.value <- mean(x, na.rm = TRUE)
   } else {
      x.tbl <- table(x, useNA = NULL)
      fill.value <- names(x.tbl)[which.max(x.tbl)]
   }
   return(fill.value)
}
fill.values <- census.income %>% 
   select(-income) %>% 
   map(fillMissingValue)
dt.fill <- census.income %>% 
   replace_na(replace = fill.values)

## Create numeric matrix using one-hot-encode
x <- model.matrix(object = income ~ .-1, data = dt.fill)
dim(x)

## Variable sparsity
zero.prop <- apply(x, 2, function(x) sum(x == 0)) / nrow(x)
g <- data_frame(x = zero.prop) %>% 
   ggplot(aes(x = x, y = ..count.., fill = ..count..)) +
   geom_histogram(bins = 50) +
   scale_fill_distiller(name = "Count", palette = "Spectral", limits = c(0, 50)) +
   guides(fill = guide_colorbar(barwidth = 0.5, barheight = 10)) +
   scale_y_continuous(breaks = seq(0, 50, by = 10), limits = c(0, 50)) +
   labs(x = "Variable sparsity", y = "Frequency") +
   theme_ipsum(axis_title_size = 12)
plot(g)

## Variable cardinality
cat.prop <- census.income %>% 
   select(-income, -which(map_lgl(., is.numeric))) %>% 
   map(function(x) as.integer(table(x))) %>% 
   reduce(c)
g <- data_frame(x = cat.prop) %>% 
   ggplot(aes(x = x, y = ..count.., fill = ..count..)) +
   geom_histogram(bins = 50) +
   scale_fill_distiller(name = "Count", palette = "Spectral", limits = c(0, 50)) +
   guides(fill = guide_colorbar(barwidth = 0.5, barheight = 10)) +
   scale_y_continuous(breaks = seq(0, 50, by = 10), limits = c(0, 50)) +
   labs(x = "Category frequency", y = "Frequency") +
   theme_ipsum(axis_title_size = 12)
plot(g)

## Feature transformation
g <- data_frame(
   x = c(dt.fill$fnlwgt, sqrt(dt.fill$fnlwgt), log(dt.fill$fnlwgt)),
   type = factor(
      rep(c("Original", "Sqrt transformation", "Log transformation"), each = nrow(dt.fill)),
      levels = c("Original", "Sqrt transformation", "Log transformation")
   )
) %>% 
   ggplot(aes(x = x, fill = type)) +
   geom_density(adjust = 2, size = 0.5, alpha = 0.8) +
   scale_fill_brewer(guide = "none", palette = "Set1") +
   facet_wrap(~type, scales = "free") +
   theme_ipsum(axis_title_size = 12) +
   labs(x = "Feature values", y = "KDE Estimates")
plot(g)

## Row stats features
col.scale <- colMaxs(abs(x))
x.sc <- sweep(x, 2, col.scale, "/")
x.rowstats <- round(cbind(
   rowMeans(x.sc),
   rowMedians(x.sc),
   rowSds(x.sc),
   rowIQRs(x.sc),
   rowQuantiles(x.sc, probs = c(0.1, 0.25, 0.75, 0.9)),
   apply(x.sc, 1, function(x.row) sum(x.row==0)),
   apply(x.sc, 1, which.max),
   apply(x.sc, 1, which.min),
   apply(census.income, 1, function(data.row) sum(is.na(data.row)))
), 6)
summary(x.rowstats)
col.vars <- apply(x.rowstats, 2, var)
x.rowstats <- x.rowstats[,-which(col.vars == 0)]
colnames(x.rowstats) <- paste0("rowstats_", 1:ncol(x.rowstats))
dim(x.rowstats)
save(x.rowstats, file = "./data/rowstats_features.rda", compress = "bzip2")

## Plot rowstats features
set.seed(2020)
g <- bind_cols(
   as.data.frame(x.rowstats),
   select(dt.fill, income)                 
) %>% 
   group_by(income) %>% 
   mutate(wt = n()) %>% 
   ungroup() %>% 
   mutate(wt = 1 - wt / n()) %>% 
   sample_n(size = 5e3, weight = wt) %>% 
   select(-wt) %>% 
   melt(id.vars = "income", variable.name = "feature") %>% 
   ggplot(aes(x = income, y = value, fill = income, color = income)) +
   geom_jitter(size = 0.3, alpha = 0.4, color = "black") +
   geom_boxplot(outlier.shape = NA, alpha = 0.6, size = 0.5) +
   scale_color_brewer(name = "Income", palette = "Set1") +
   scale_fill_brewer(name = "Income", palette = "Set1") +
   facet_wrap(~feature, ncol = 3, scales = "free_y") +
   theme_ipsum(axis_title_size = 0)
g

## Clustering features
set.seed(2020)
cl.model <- dbscan(x.sc, eps = 0.5, minPts = 128)
cl.features <- data_frame(
   cluster = factor(cl.model$cluster, levels = sort(unique(cl.model$cluster))),
   class = dt.fill$income
) %>% 
   mutate(
      pos_class_prop = sum(class == ">50k") / n(),
      neg_class_prop = sum(class == "<=50k") / n()
   ) %>% 
   group_by(cluster) %>% 
   mutate(
      pos_cluster_prop = sum(class == ">50k") / n() / pos_class_prop,
      neg_cluster_prop = sum(class == "<=50k") / n() / neg_class_prop
   ) %>%
   ungroup() %>% 
   mutate(
      pos_cluster_prop = pos_cluster_prop / (pos_cluster_prop + neg_cluster_prop),
      neg_cluster_prop = neg_cluster_prop / (pos_cluster_prop + neg_cluster_prop)
   )
g <- ggplot(cl.features, aes(x = pos_cluster_prop, fill = class)) +
   geom_density(size = 0.5, adjust = 5, color = "black", alpha = 0.8) +
   scale_fill_brewer(name = "Income", palette = "Set1") +
   scale_x_continuous(breaks = seq(0, 1, by = 0.25), limits = c(0, 1)) +
   labs(x = "Positive class proportion inside each cluster", y = "KDE Estimates") +
   theme_ipsum(axis_title_size = 12)
plot(g)
x.cluster <- cbind(
   table(1:nrow(cl.features), cl.features$cluster), cl.features$pos_cluster_prop
)
colnames(x.cluster) <- paste0("cluster_", 1:ncol(x.cluster))
rownames(x.cluster) <- NULL   
save(x.cluster, file = "./data/cluster_features.rda", compress = "bzip2")

## PCA features
pca.model <- prcomp(x.sc, retx = FALSE, center = TRUE, scale. = FALSE)
x.pca <- predict(pca.model, x.sc)[,1:3]
colnames(x.pca) <- paste0("pca_", 1:ncol(x.pca))
exp.var <- cumsum(pca.model$sdev) / sum(pca.model$sdev)
g <- data_frame(
   var = exp.var,
   npcs = 1:length(exp.var)
) %>% 
   ggplot(aes(x = npcs, y = var, fill = var)) +
   geom_col(width = 0.8, size = 0.1, alpha = 0.8, color = "white") +
   scale_fill_distiller(guide = "none", palette = "Spectral") + 
   geom_hline(yintercept = 0.8, size = 0.5, color = "gray40", linetype = "dashed") +
   geom_vline(aes(xintercept = npcs[which(var >= 0.8)[1]]), size = 0.5, 
              color = "gray40", linetype = "dashed") +
   scale_x_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
   scale_y_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
   labs(x = "Number of PCs", y = "Explained Variance") +
   theme_ipsum(axis_title_size = 12)
plot(g)
save(x.pca, file = "./data/pca_features.rda", compress = "bzip2")

## Plot PCA features
g <- data_frame(
   pc1 = x.pca[, 1], pc2 = x.pca[, 2], y = dt.fill$income
) %>% 
   ggplot(aes(x = pc1, y = pc2, color = y)) +
   geom_point(size = 1, alpha = 0.8) +
   scale_color_brewer(name = "Income", palette = "Set1") +
   labs(x = "PC1", y = "PC2") +
   theme_ipsum(axis_title_size = 12)
plot(g)

## LDA features
lda.model <- lda(x = x.sc, grouping = dt.fill$income)
x.lda <- predict(lda.model, x.sc)$x
colnames(x.lda) <- paste0("lda_", 1:ncol(x.lda))
save(x.lda, file = "./data/lda_features.rda", compress = "bzip2")

## Plot LDA features
g <- data_frame(
   x = x.lda[, 1], y = dt.fill$income
) %>% 
   ggplot(aes(x = x, fill = y)) +
   geom_density(size = 0.5, adjust = 2, color = "black", alpha = 0.8) +
   scale_fill_brewer(name = "Income", palette = "Set1") +
   # scale_x_continuous(breaks = seq(0, 1, by = 0.25), limits = c(0, 1)) +
   labs(x = "LDA Feature", y = "KDE Estimates") +
   theme_ipsum(axis_title_size = 12)
plot(g)
   
## t-SNE Features
set.seed(2020)
x.tsne <- Rtsne(x, check_duplicates = FALSE, pca = TRUE, initial_dims = 16, 
               perplexity = 15, theta = 0.5, dims = 3, verbose = TRUE, 
               max_iter = 500)$Y
colnames(x.tsne) <- paste0("tsne_", 1:ncol(x.tsne))
save(x.tsne, file = "./data/tsne_features.rda", compress = "bzip2")

## Plot t-SNE features
p <- plot_ly(x = x.tsne[,1], y = x.tsne[,2], z = x.tsne[,3], 
             color = dt.fill$income, colors = c('black', 'red'), 
             opacity = 0.6, marker = list(size = 3)) %>%
   add_markers() %>%
   layout(scene = list(
      xaxis = list(title = 'tSNE 1'), yaxis = list(title = 'tSNE 2'), 
      zaxis = list(title = 'tSNE 3')
   ))
p

## Autoencoder features
h2o.init(nthreads = 4, max_mem_size = '4G')
h2o.removeAll()
data.hex <- as.h2o(as.data.frame(x.sc), destination_frame = "data_hex")
dl.model <- h2o.deeplearning(x = colnames(x.sc), training_frame = data.hex, 
                             activation = "Tanh", epochs = 50, seed = 2020,
                             hidden=c(128, 64, 3, 64, 128), autoencoder = TRUE)
x.autoencoder <- h2o.deepfeatures(dl.model, data = data.hex, layer = 3) %>% 
   as.data.frame() %>% 
   data.matrix()
h2o.shutdown(prompt = FALSE)
colnames(x.autoencoder) <- paste0("autoencoder_", 1:ncol(x.autoencoder))
save(x.autoencoder, file = "./data/autoencoder_features.rda", compress = "bzip2")

## Plot autoencoder features
p <- plot_ly(x = x.autoencoder[,1], y = x.autoencoder[,2], z = x.autoencoder[,3], 
             color = dt.fill$income, colors = c('black', 'red'), 
             opacity = 0.6, marker = list(size = 3)) %>%
   add_markers() %>%
   layout(scene = list(
      xaxis = list(title = 'DL 1'), yaxis = list(title = 'DL 2'), 
      zaxis = list(title = 'DL 3')
   ))
p

## KNN Features with toy data
x.circles <- data.matrix(concentric.circles[,-3])
g <- knnDecision(xtr = x.circles, ytr = concentric.circles$class, xte = x.circles, 
                 yte = concentric.circles$class, k = 3) +
   theme_ipsum(axis_title_size = 12)
plot(g)
new.x <- knnExtract(xtr = x.circles, ytr = concentric.circles$class, 
                    xte = x.circles, k = 1)
g <- knnDecision(xtr = new.x$new.tr, ytr = concentric.circles$class, 
                 xte = new.x$new.tr, yte = concentric.circles$class, k = 3) +
   theme_ipsum(axis_title_size = 12)
plot(g)

## KNN Features
x.knn <- knnExtract(xtr = x.sc, ytr = dt.fill$income, xte = x.sc, k = 1, 
                    nthread = 4)$new.tr
g <- data_frame(
   x1 = x.knn[, 1], x2 = x.knn[, 2], y = dt.fill$income
) %>% 
   ggplot(aes(x = x1, y = x2, color = y)) +
   geom_point(size = 1, alpha = 0.8) +
   scale_color_brewer(name = "Income", palette = "Set1") +
   labs(x = "KNN Feature 1", y = "KNN Feature 2") +
   theme_ipsum(axis_title_size = 12)
plot(g)
g <- knnDecision(xtr = x.knn, ytr = dt.fill$income, xte = x.knn, 
                 yte = dt.fill$income, k = 3) +
   theme_ipsum(axis_title_size = 12)
plot(g)

## Generate more KNN features
x.knn <- knnExtract(xtr = x.sc, ytr = dt.fill$income, xte = x.sc, k = 5, 
                    nthread = 4)$new.tr
colnames(x.knn) <- paste0("knn_", 1:ncol(x.knn))
save(x.knn, file = "./data/knn_features.rda", compress = "bzip2")

## Plot 3D KNN features
p <- plot_ly(x = x.knn[, 1], y = x.knn[, 5], z = x.knn[, 10], 
             color = dt.fill$income, colors = c('black', 'red'), 
             opacity = 0.6, marker = list(size = 3)) %>%
   add_markers() %>%
   layout(scene = list(
      xaxis = list(title = 'KNN 1'), yaxis = list(title = 'KNN 5'), 
      zaxis = list(title = 'KNN 10')
   ))
p

## Train xgboost with original features
dtrain <- xgb.DMatrix(
   Matrix(x.sc, sparse = TRUE),
   label = as.integer(dt.fill$income) - 1
)
xgb.params <- list(
   "booster" = "gbtree",
   "eta" = 0.05,
   "max_depth" = 4,
   "subsample" = 0.632,
   "colsample_bytree" = 0.4,
   "colsample_bylevel" = 0.6,
   "min_child_weight" = 1,
   "gamma" = 0,
   "lambda" = 0,
   "alpha" = 0,
   "objective" = "binary:logistic",
   "eval_metric" = "auc",
   "silent" = 1,
   "nthread" = 4,
   "num_parallel_tree" = 5
)
set.seed(2020)
cv.out <- xgb.cv(params = xgb.params, data = dtrain, nrounds = 1.5e3,
                 metrics = list('error'), nfold = 5, prediction = FALSE, 
                 verbose = TRUE, showsd = FALSE, print.every.n = 10, 
                 early.stop.round = 10, maximize = TRUE)
max(cv.out$train.auc.mean)
min(cv.out$train.error.mean)
max(cv.out$test.auc.mean)
min(cv.out$test.error.mean)
which.max(cv.out$test.auc.mean)
xgb.model <- xgb.train(data = dtrain, params = xgb.params, 
                       nrounds = 750)
var.imp <- xgb.importance(colnames(x.sc), model = xgb.model) %>% 
   mutate(Feature = gsub('[0-9]+', '', Feature)) %>%
   group_by(Feature) %>% 
   summarise(Importance = quantile(Gain, 0.9)) %>% 
   ungroup() %>% 
   arrange(desc(Importance)) %>% 
   mutate(Importance = round(100 * Importance / sum(Importance), 2))

## Train xgboost with new features
load("data/rowstats_features.rda")
load("data/cluster_features.rda")
load("data/pca_features.rda")
load("data/lda_features.rda")
load("data/tsne_features.rda")
load("data/autoencoder_features.rda")
load("data/knn_features.rda")
new.x <- cbind(
   x.rowstats, x.cluster, x.pca, x.lda, x.tsne, x.autoencoder, x.knn
)
dtrain <- xgb.DMatrix(
   Matrix(new.x, sparse = TRUE),
   label = as.integer(dt.fill$income) - 1
)
set.seed(2020)
cv.out <- xgb.cv(params = xgb.params, data = dtrain, nrounds = 1.5e3,
                 metrics = list('error'), nfold = 5, prediction = FALSE, 
                 verbose = TRUE, showsd = FALSE, print.every.n = 10, 
                 early.stop.round = 10, maximize = TRUE)
max(cv.out$train.auc.mean)
min(cv.out$train.error.mean)
max(cv.out$test.auc.mean)
min(cv.out$test.error.mean)
which.max(cv.out$test.auc.mean)
xgb.model <- xgb.train(data = dtrain, params = xgb.params, 
                       nrounds = 750)
var.imp <- xgb.importance(colnames(new.x), model = xgb.model) %>% 
   mutate(Feature = gsub('[0-9]:', '', Feature)) %>%
   group_by(Feature) %>% 
   summarise(Importance = quantile(Gain, 0.9)) %>% 
   ungroup() %>% 
   arrange(desc(Importance)) %>% 
   mutate(Importance = round(100 * Importance / sum(Importance), 2))

## Train xgboost with all features
all.x <- cbind(x.sc, new.x)
dtrain <- xgb.DMatrix(
   Matrix(all.x, sparse = TRUE),
   label = as.integer(dt.fill$income) - 1
)
set.seed(2020)
cv.out <- xgb.cv(params = xgb.params, data = dtrain, nrounds = 1.5e3,
                 metrics = list('error'), nfold = 5, prediction = FALSE, 
                 verbose = TRUE, showsd = FALSE, print.every.n = 10, 
                 early.stop.round = 10, maximize = TRUE)
max(cv.out$train.auc.mean)
min(cv.out$train.error.mean)
max(cv.out$test.auc.mean)
min(cv.out$test.error.mean)
which.max(cv.out$test.auc.mean)
xgb.model <- xgb.train(data = dtrain, params = xgb.params, 
                       nrounds = 500)
var.imp <- xgb.importance(colnames(all.x), model = xgb.model) %>% 
   mutate(Feature = gsub('[0-9]:', '', Feature)) %>%
   group_by(Feature) %>% 
   summarise(Importance = quantile(Gain, 0.9)) %>% 
   ungroup() %>% 
   arrange(desc(Importance)) %>% 
   mutate(Importance = round(100 * Importance / sum(Importance), 2))

## Train GLM with original features
orig.hex <- as.h2o(cbind.data.frame(as.data.frame(x.sc), income = dt.fill$income), 
                   destination_frame = "orig_hex")
glm.model <- h2o.glm(x = colnames(x.sc), y = "income", training_frame = orig.hex, 
                     nfolds = 5, seed = 2020, family = "binomial", alpha = 0, 
                     lambda = 0, solver = "IRLSM", standardize = TRUE, 
                     intercept = TRUE, max_iterations = 1e3)
h2o.auc(glm.model@model$cross_validation_metrics)
h2o.confusionMatrix(glm.model@model$cross_validation_metrics)

## Train GLM with new features
new.hex <- as.h2o(cbind.data.frame(as.data.frame(new.x), income = dt.fill$income), destination_frame = "new_hex")
glm.model <- h2o.glm(x = colnames(new.x), y = "income", training_frame = new.hex, 
                     nfolds = 5, seed = 2020, family = "binomial", alpha = 0, 
                     lambda = 0, solver = "IRLSM", standardize = TRUE, 
                     intercept = TRUE, max_iterations = 1e3)
h2o.auc(glm.model@model$cross_validation_metrics)
h2o.confusionMatrix(glm.model@model$cross_validation_metrics)

## Train GLM with all features
all.hex <- as.h2o(cbind.data.frame(as.data.frame(all.x), income = dt.fill$income), destination_frame = "all_hex")
glm.model <- h2o.glm(x = colnames(all.x), y = "income", training_frame = all.hex, 
                     nfolds = 5, seed = 2020, family = "binomial", alpha = 0, 
                     lambda = 0, solver = "IRLSM", standardize = TRUE, 
                     intercept = TRUE, max_iterations = 1e3)
h2o.auc(glm.model@model$cross_validation_metrics)
h2o.confusionMatrix(glm.model@model$cross_validation_metrics)

## Train GLM with features selection
glm.model <- h2o.glm(x = colnames(all.x), y = "income", training_frame = all.hex, 
                     nfolds = 5, seed = 2020, family = "binomial", alpha = 1, 
                     lambda_search = TRUE, nlambdas = 50, 
                     max_active_predictors = 75, solver = "IRLSM", 
                     standardize = TRUE, intercept = TRUE, max_iterations = 1e3)
h2o.auc(glm.model@model$cross_validation_metrics)
h2o.confusionMatrix(glm.model@model$cross_validation_metrics)
head(h2o.varimp(glm.model), 20)
tail(h2o.varimp(glm.model), 20)
