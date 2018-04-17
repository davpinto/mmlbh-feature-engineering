## Load required packages
library(readr)
library(magrittr)
library(dplyr)
library(purrr)

## Import column names
data.names <- read_delim("./data/census_income_names.txt", delim = ":", 
                         col_names = c("name", "descrption"), trim_ws = TRUE) %>% 
   mutate(name = gsub(" ", "_", tolower(trimws(name))), name = gsub("-", "_", name))

## Import dataset
census.income <- read_delim("./data/census_income_data.txt", delim = ",", 
                            col_names = FALSE, trim_ws = TRUE, na = c("?", "")) %>% 
   set_names(c(data.names$name, "income"))

## Format categorical variables
formatCatVar <- function(x) {
   x <- gsub(" ", "_", tolower(trimws(x)))
   x <- gsub("-", "_", x)
   x.levels <- sort(unique(x))
   if(any(is.na(x.levels))) {
      x.levels <- x.levels[-which(is.na(x.levels))]
   }
   x <- factor(x, levels = x.levels)
   return(x)
}
cat.vars <- which(map_lgl(census.income, is.character))
census.income <- census.income %>% 
   mutate_each(funs(formatCatVar), cat.vars)

## Save dataset
save(census.income, file = "./data/census_income.rda", compress = "bzip2")
