## Load required packages
library(magrittr)
library(purrr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(reshape2)
library(plotly)
library(GGally)
library(Matrix)
library(matrixStats)
library(cluster)
library(h2o)
library(Rtsne)
library(fastknn)
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
cl.model <- clara(x.sc, k = 6, stand = FALSE, medoids.x = FALSE, 
                  keep.data = FALSE)

library(fastknn)

x <- data.matrix(concentric.circles[,-3])
new.x <- knnExtract(x, concentric.circles$class, x)

plot(x, col = concentric.circles$class, pch = 20)
plot(new.x$new.tr, col = concentric.circles$class, pch = 20)
