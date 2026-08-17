library(haven)
library(tidyverse)
library(ggplot2)
library(readxl)

# Demographic data
dat <-
  read_dta(
    "../data/PowerED RL Data_Table 1.dta"
  ) %>%
  mutate(reward = as.numeric(reward))
demo_dat <- dat %>%
  mutate(
    race = case_when(
      dem4_3 == 1 ~ "asian",
      dem4_4 == 1 ~ "Native American",
      dem4_7 == 1 ~ "More than 1",
      dem4_1 == 1 ~ "black",
      dem4_2 == 1 ~ "white",
      TRUE ~ "other"
    ),
    education = case_match(
      dem8,
      c(1, 2) ~ "less than high school",
      c(3, 4) ~ "high school",
      c(5, 6) ~ "college"
    ),
    phq9_score = phq9_1 + phq9_2 + phq9_3 + phq9_4 + phq9_5 + phq9_6 + phq9_7 + phq9_8 + phq9_9,
    depression = case_match(phq9_score, 0:4 ~ "none", 5:9 ~ "mild", 10:27 ~ "mod/severe")
  ) %>%
  select(pt_id,
         dem1,
         dem2,
         dem3,
         dem7,
         pt1,
         starts_with("comm"),
         depression,
         race,
         education) %>%
  rename(
    age = "dem1",
    sex = "dem2",
    ethnicity = "dem3",
    employed = "dem7"
  ) %>%
  mutate(
    comm_4 = na_if(comm_4,-99),
    comm_3 = na_if(comm_3,-99),
    ethnicity = na_if(ethnicity, 888),
    ethnicity = na_if(ethnicity,-99)
  ) %>%
  unique() %>%
  drop_na() %>%
  mutate(comm_score = comm_1 + comm_2 + comm_3 + comm_4 + comm_5 + comm_6 + comm_7 + comm_8) %>%
  drop_na() %>%
  select(-c(comm_1, comm_2, comm_3, comm_4, comm_5, comm_6, comm_7, comm_8))

# Weekly data
dat <-
  read_excel("../data/PowerED data.xlsx")
weekly_dat <- dat %>% select(-c(fid))

# Cmobine weekly data and demo data
dat <-
  weekly_dat %>% inner_join(demo_dat) ## remove 17 people from weekly data due to no demo info


# Remove week 13 data from some people
dat <- dat %>% filter(week != 13)

# Remove 6 people with less than half of the weeks
dat <-
  dat %>% filter(!(
    pt_id %in% (dat %>% group_by(pt_id) %>% count() %>% filter(n < 6) %>% pull(pt_id))
  ))

dat %>%
  select(-c(comm)) %>%
  write_csv(
    "../data/processed_powered_data.csv"
  )
