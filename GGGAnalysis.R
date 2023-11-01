library(tidyverse)
library(vroom)
library(tidymodels)

gggMissing <- vroom("./GhostsGhoulsandGoblins/trainWithMissingValues.csv")
gggTrain <- vroom("./GhostsGhoulsandGoblins/train.csv")
gggTest <- vroom("./GhostsGhoulsandGoblins/test.csv")

my_recipe <- recipe(type~., data = gggMissing) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(all_numeric_predictors())) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(all_numeric_predictors())) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(all_numeric_predictors()))
  
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = gggMissing)

rmse_vec(gggTrain[is.na(gggMissing)], baked[is.na(gggMissing)])
