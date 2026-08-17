library(mice)
library(tidyverse)

#
dat <-
  read_csv(
    "../data/processed_powered_data.csv"
  ) %>%
  select(-c(month))
dat <- dat %>%
  mutate(rldecision = replace_na(rldecision, 0)) %>% ## rldecision == 0 means no trt
  mutate(
    sex = factor(sex),
    ethnicity = factor(ethnicity),
    employed = factor(employed),
    pt1 = factor(pt1),
    rldecision = factor(rldecision),
    ivr_cluster = factor(ivr_cluster)
  )
bl_dat = dat %>% select(
  -c(
    reward,
    weekly_pain_score,
    weekly_pain_interference,
    week,
    calendar_week,
    rldecision
  )
) %>%
  distinct()
weekly_dat = dat %>% select(
  pt_id,
  week,
  calendar_week,
  rldecision,
  reward,
  weekly_pain_score,
  weekly_pain_interference
) %>%
  distinct()

cur_seed = 1
set.seed(cur_seed)

impute_data = tibble()
pt_ids <- unique(dat$pt_id)

# impute week = 1
per_week_data = expand_grid(pt_id = pt_ids, week = 1)
per_week_data = per_week_data %>% left_join(weekly_dat) %>% left_join(bl_dat)
for (a in c(1, 2, 3)) {
  if (a == 1) {
    ## i need to include a == 0 to a == 1, since we have few observations with a == 0
    per_week_data_action = per_week_data %>% filter(rldecision == 1 |
                                                      rldecision == 0)
  }
  else{
    per_week_data_action = per_week_data %>% filter(rldecision == a)
  }
  mice_input = per_week_data_action %>% select(-c(pt_id, week))
  mice_output = complete(mice(
    mice_input,
    m = 1,
    method = 'pmm',
    printFlag = FALSE
  ), 1) %>% tibble()
  mice_output = per_week_data_action %>% select(pt_id, week) %>% bind_cols(mice_output)
  impute_data = impute_data %>% bind_rows(mice_output)
}

# impute week = 2, ..., 12
for (i in 2:12) {
  per_week_data = expand_grid(pt_id = pt_ids, week = i)
  per_week_data = per_week_data %>% left_join(weekly_dat)
  history_data = impute_data %>%
    filter(week == i - 1) %>%
    select(-c(week, calendar_week)) %>%
    rename(
      rldecision_m1 = "rldecision",
      reward_m1 = "reward",
      weekly_pain_score_m1 = "weekly_pain_score",
      weekly_pain_interference_m1 = "weekly_pain_interference"
    )
  # impute rldecision at week == i
  mice_input = per_week_data %>% select(pt_id, rldecision) %>%
    inner_join(history_data) %>%
    select(-c(pt_id))
  mice_output = complete(mice(
    mice_input,
    m = 1,
    method = 'pmm',
    printFlag = FALSE
  ), 1) %>% tibble()
  per_week_data = per_week_data %>% select(-c(rldecision)) %>%
    bind_cols(mice_output)
  
  # impute reward, weekly_pain_score, weekly_pain_interference
  for (a in c(1, 2, 3)) {
    print(paste(a, i))
    if (a == 1) {
      ## i need to include a == 0 to a == 1, since we have few observations with a == 0
      per_week_data_action = per_week_data %>% filter(rldecision == 1 |
                                                        rldecision == 0)
    }
    else{
      per_week_data_action = per_week_data %>% filter(rldecision == a)
    }
    mice_input = per_week_data_action %>% select(-c(pt_id, week))
    if (nrow(mice_input) > 1) {
      mice_output = complete(mice(
        mice_input,
        m = 1,
        method = 'pmm',
        printFlag = FALSE
      ), 1) %>% tibble()
    }
    else{
      mice_output = mice_input
    }
    mice_output = per_week_data_action %>% select(pt_id, week) %>% bind_cols(mice_output) %>% select(!ends_with("m1"))
    impute_data = impute_data %>% bind_rows(mice_output)
  }
}

impute_data %>% write_csv(
  "../data/impute_data.csv"
)
