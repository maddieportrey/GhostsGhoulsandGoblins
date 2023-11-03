library(tidyverse)
library(vroom)
library(tidymodels)

gggMissing <- vroom("./trainWithMissingValues.csv")
gggTrain <- vroom("./train.csv")
gggTest <- vroom("./test.csv")

my_recipe <- recipe(type~., data = gggMissing) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(all_numeric_predictors())) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(all_numeric_predictors())) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(all_numeric_predictors())) %>%
  step_rm(id) %>%
  step_dummy(all_nominal_predictors())
  
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = gggMissing)

rmse_vec(gggTrain[is.na(gggMissing)], baked[is.na(gggMissing)])

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

ggg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(baked)-1))),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(gggTrain, v = 10, repeats = 1)

CV_results <- ggg_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- ggg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = gggTrain)

ggg_preds <- final_wf %>% predict(new_data = gggTest, type = "class")

preds <- cbind(gggTest$id, ggg_preds)
colnames(preds) <- c("id","type")
preds <- as.data.frame(preds)
vroom_write(preds, "ggg_predictions.csv", ",")
