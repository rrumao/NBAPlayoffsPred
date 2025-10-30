# Load in necessary libraries
library(readxl)
library(dplyr)
library(glmnet)
library(ggplot2)
library(tidyverse)
library(glmnet)
library(randomForest)
library(tree)

# Read in the player and team data
player.data = read_excel('./PlayerStats.xlsx')
team.data = read_excel('./TeamStatsFinal.xlsx', sheet = 'Converted', range = 'A1:Z151')

# Drop columns from team data that we won't need for the analysis

###
# Rank is just a column to identify each team.
# Division contributes quite little to the data, Conf covers most of the information.
# GP is 82 for all teams across each season, so it is just constant and hence, won't be used in our analysis
# STRK is a metric added by the data source that we suspect won't contribute heavily.
###
teams = team.data %>% 
  dplyr::select(-RANK, -DIVISION, -GP, -STRK)

###
# Rank is just a column to identify each player.
# TOpG, while useful, is not neccesary as we have TO%.
# P+R, P+A, P+R+A are not useful as we have each of the metrics individually in the data.
# VI is a metric added by the data source, won't contribute much as the information it provides is covered by other features.
###
players = player.data %>% 
  dplyr::select(-RANK, -TOpG, -'P+R', -'P+A', -'P+R+A', -VI)


# Clean the player data set to get rid of players who might skew the features and introduce bias
players = players %>% 
  filter(GP >= 10, MpG >= 5)


# Player Aggregates
aggregates = players %>% 
  group_by(SEASON, TEAM) %>% 
  summarise(avg_TS = weighted.mean(`TS%`, w = MpG),
            avg_TO   = weighted.mean(`TO%`, w = MpG),
            avg_USG   = weighted.mean(`USG%`, w = MpG),
            avg_ORtg = weighted.mean(ORtg, w = MpG),
            avg_DRtg = weighted.mean(DRtg, w = MpG),
            avg_APG = weighted.mean(ApG, w = MpG),
            avg_RPG = weighted.mean(RpG, w = MpG),
            avg_AGE = weighted.mean(AGE, w = MpG),
            avg_SPG  = weighted.mean(SpG, w = MpG),
            avg_EFG   = weighted.mean(`eFG%`, w = MpG),
            avg_BPG  = weighted.mean(BpG, w = MpG),
            .groups = "drop")

# Join Player and Team data

data = merge(aggregates, teams, by=c('SEASON', 'TEAM'))

# Train-Test split

# Train-test split by season
train = data %>% filter(SEASON %in% 2020:2024)
test  = data %>% filter(SEASON == 2025)

# Linear Regression
lm.model = lm(PLAYOFFS ~ avg_ORtg + avg_DRtg + avg_TS + avg_TO + 
                                    avg_APG + avg_RPG + avg_SPG + avg_BPG + 
                                    avg_AGE + PACE + SoS + eDIFF + CONS, data = train)

summary(lm.model)
test$pred.lm = predict(lm.model, newdata = test)

# Linear Regression with Interactions
lm.interact = lm(PLAYOFFS ~ avg_ORtg * avg_DRtg + avg_TS * CONS + avg_TO * CONS
                 + eDIFF * CONS, data = train)

summary(lm.interact)
test$pred.lm.interact = predict(lm.interact, newdata = test)

# LASSO Regression
x.train = model.matrix(PLAYOFFS ~ (avg_ORtg + avg_DRtg + avg_TS + avg_TO + 
                                    avg_APG + avg_RPG + avg_SPG + avg_BPG + 
                                    avg_AGE + PACE + SoS + eDIFF + CONS)^2, data = train)[, -1]

y.train = train$PLAYOFFS
x.test =  model.matrix(PLAYOFFS ~ (avg_ORtg + avg_DRtg + avg_TS + avg_TO + 
                                   avg_APG + avg_RPG + avg_SPG + avg_BPG + 
                                   avg_AGE + PACE + SoS + eDIFF + CONS)^2, data = test)[, -1]

y.test = test$PLAYOFFS

set.seed(123)

lasso.1 = cv.glmnet(x.train, y.train, alpha = 1, nfolds = 5)
lambda = lasso.1$lambda.min
lasso.model = glmnet(x.train, y.train, alpha = 1, lambda = lambda)
pred.lasso = predict(lasso.model, newx = x.test, s = "lambda.min")
test$pred.lasso = pred.lasso


lambda.1se = lasso.1$lambda.1se
lasso.model.1se = glmnet(x.train, y.train, alpha = 1, lambda = lambda.1se)
pred.lasso = predict(lasso.model.1se, newx = x.test, s = "lambda.1se")
test$pred.lasso.2 = pred.lasso

# Regression Tree
set.seed(123)

tree.model = tree(PLAYOFFS ~ avg_ORtg + avg_DRtg + avg_TS + avg_TO + 
                    avg_APG + avg_RPG + avg_SPG + avg_BPG + 
                    avg_AGE + PACE + SoS + eDIFF + CONS, data = train)
plot(tree.model)
text(tree.model)
cv_tree = cv.tree(tree.model)
plot(cv_tree$size, cv_tree$dev, type = "b")
best.size = cv_tree$size[which.min(cv_tree$dev)]
pruned.tree = prune.tree(tree.model, best = best.size)
plot(pruned.tree)
text(pruned.tree)
test$pred.tree = predict(pruned.tree, newdata = test)

# Random Forest
set.seed(123)

rf.model = randomForest(PLAYOFFS ~ avg_ORtg + avg_DRtg + avg_TS + avg_TO + 
                          avg_APG + avg_RPG + avg_SPG + avg_BPG + 
                          avg_AGE + PACE + SoS + eDIFF + CONS, data = train, ntree = 500,
                        mtry = 7)
test$pred.rf = predict(rf.model, newdata = test)

# Evaluation
# OOS Accuracy and Root Mean Squared Error
# Accuracy within each playoff rank as well, i.e, how good was the model at
# predicting the champion, runner-up, conference finalists, play-in qualifiers etc.

test = test %>%
  arrange(desc(pred.lm)) %>%
  mutate(
    playoff.rank = row_number(),
    playoffs.pred.lm = case_when(
      playoff.rank == 1 ~ 6,
      playoff.rank == 2 ~ 5,      
      playoff.rank %in% 3:4 ~ 4,      
      playoff.rank %in% 5:8 ~ 3,      
      playoff.rank %in% 9:16 ~ 2,    
      playoff.rank %in% 17:20 ~ 1,    
      TRUE ~ 0                        
    )
  )

accuracy.lm = mean(test$playoffs.pred.lm == test$PLAYOFFS)
accuracy.lm

accuracy.rank.lm = test %>%
  group_by(PLAYOFFS) %>%
  summarise(
    n = n(),
    correct = sum(playoffs.pred.lm == PLAYOFFS),
    accuracy = round(correct / n, 2)
  ) %>%
  arrange(desc(accuracy))
accuracy.rank.lm$name = 'Linear Reg'
accuracy.rank.lm

rmse.lm = sqrt(mean((test$PLAYOFFS - test$pred.lm)^2))
rmse.lm

test = test %>%
  arrange(desc(pred.lm.interact)) %>%
  mutate(
    playoff.rank = row_number(),
    playoffs.pred.lm.interact = case_when(
      playoff.rank == 1 ~ 6,
      playoff.rank == 2 ~ 5,
      playoff.rank %in% 3:4 ~ 4,
      playoff.rank %in% 5:8 ~ 3,
      playoff.rank %in% 9:16 ~ 2,
      playoff.rank %in% 17:20 ~ 1,
      TRUE ~ 0
    )
  )

accuracy.lm.interact = mean(test$playoffs.pred.lm.interact == test$PLAYOFFS)
accuracy.lm.interact 

accuracy.rank.interact = test %>%
  group_by(PLAYOFFS) %>%
  summarise(
    n = n(),
    correct = sum(playoffs.pred.lm.interact == PLAYOFFS),
    accuracy = round(correct / n, 2)
  ) %>%
  arrange(desc(accuracy))
accuracy.rank.interact$name = 'Linear Reg with Interactions'
accuracy.rank.interact

rmse.lm.interact = sqrt(mean((test$PLAYOFFS - test$pred.lm.interact)^2))
rmse.lm.interact


test = test %>%
  arrange(desc(pred.lasso)) %>%
  mutate(
    playoffs.rank = row_number(),
    playoffs.pred.lasso = case_when(
      playoffs.rank == 1 ~ 6,          
      playoffs.rank == 2 ~ 5,          
      playoffs.rank %in% 3:4 ~ 4,      
      playoffs.rank %in% 5:8 ~ 3,     
      playoffs.rank %in% 9:16 ~ 2,    
      playoffs.rank %in% 17:20 ~ 1,   
      TRUE ~ 0                       
    )
  )


accuracy.lasso = mean(test$playoffs.pred.lasso == test$PLAYOFFS)
accuracy.lasso

accuracy.rank.lasso = test %>%
  group_by(PLAYOFFS) %>%
  summarise(
    n = n(),
    correct = sum(playoffs.pred.lasso == PLAYOFFS),
    accuracy = round(correct / n, 2)
  ) %>%
  arrange(desc(accuracy))

accuracy.rank.lasso$name = 'Lasso Reg (Min Lambda)'
accuracy.rank.lasso

rmse.lasso = sqrt(mean((test$PLAYOFFS - test$pred.lasso)^2))
rmse.lasso

test = test %>%
  arrange(desc(pred.lasso.2)) %>%
  mutate(
    playoffs.rank = row_number(),
    playoffs.pred.lasso.2 = case_when(
      playoffs.rank == 1 ~ 6,          
      playoffs.rank == 2 ~ 5,          
      playoffs.rank %in% 3:4 ~ 4,      
      playoffs.rank %in% 5:8 ~ 3,     
      playoffs.rank %in% 9:16 ~ 2,    
      playoffs.rank %in% 17:20 ~ 1,   
      TRUE ~ 0                       
    )
  )

accuracy.lasso.1se = mean(test$playoffs.pred.lasso.2 == test$PLAYOFFS)
accuracy.lasso.1se

accuracy.rank.lasso.2 = test %>%
  group_by(PLAYOFFS) %>%
  summarise(
    n = n(),
    correct = sum(playoffs.pred.lasso.2 == PLAYOFFS),
    accuracy = round(correct / n, 2)
  ) %>%
  arrange(desc(accuracy))

accuracy.rank.lasso.2$name = 'Lasso Reg (1SE)'
accuracy.rank.lasso.2

rmse.lasso.1se = sqrt(mean((test$PLAYOFFS - test$pred.lasso.2)^2))
rmse.lasso.1se

set.seed(123)
test$pred.tree.jitter = jitter(test$pred.tree)

test = test %>%
  arrange(desc(pred.tree.jitter)) %>%
  mutate(
    playoff.rank = row_number(),
    playoffs.pred.tree = case_when(
      playoff.rank == 1 ~ 6,
      playoff.rank == 2 ~ 5,     
      playoff.rank %in% 3:4 ~ 4,      
      playoff.rank %in% 5:8 ~ 3,      
      playoff.rank %in% 9:16 ~ 2,    
      playoff.rank %in% 17:20 ~ 1,    
      TRUE ~ 0                      
    )
  )

accuracy.tree = mean(test$playoffs.pred.tree == test$PLAYOFFS)
accuracy.tree

accuracy.rank.tree = test %>%
  group_by(PLAYOFFS) %>%
  summarise(
    n = n(),
    correct = sum(playoffs.pred.tree == PLAYOFFS),
    accuracy = round(correct / n, 2)
  ) %>%
  arrange(desc(accuracy))

accuracy.rank.tree$name = 'Regression Tree'
accuracy.rank.tree

rmse.tree = sqrt(mean((test$PLAYOFFS - test$pred.tree)^2))
rmse.tree

test = test %>%
  arrange(desc(pred.rf)) %>%
  mutate(
    playoff.rank = row_number(),
    playoffs.pred.rf = case_when(
      playoff.rank == 1 ~ 6,
      playoff.rank == 2 ~ 5,     
      playoff.rank %in% 3:4 ~ 4,      
      playoff.rank %in% 5:8 ~ 3,      
      playoff.rank %in% 9:16 ~ 2,    
      playoff.rank %in% 17:20 ~ 1,    
      TRUE ~ 0                      
    )
  )

accuracy.rf = mean(test$playoffs.pred.rf == test$PLAYOFFS)
accuracy.rf

accuracy.rank.rf = test %>%
  group_by(PLAYOFFS) %>%
  summarise(
    n = n(),
    correct = sum(playoffs.pred.rf == PLAYOFFS),
    accuracy = round(correct / n, 2)
  ) %>%
  arrange(desc(accuracy))

accuracy.rank.rf$name = 'Random Forest'
accuracy.rank.rf

rmse.rf = sqrt(mean((test$PLAYOFFS - test$pred.rf)^2))
rmse.rf

metrics = data.frame(Model = c('Linear Reg', 'Linear Reg with Interactions',
                               'Lasso Reg (Min Lambda)', 'Lasso Reg (1SE)',
                               'Regression Tree', 'Random Forest'),
                     RMSE = c(rmse.lm, rmse.lm.interact, rmse.lasso, rmse.lasso.1se,
                              rmse.tree, rmse.rf),
                     Accuracy = c(accuracy.lm, accuracy.lm.interact, accuracy.lasso,
                                  accuracy.lasso.1se, accuracy.tree, accuracy.rf))
metrics

ggplot(metrics, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", width = 0.5) +
  ggtitle('Root Mean Squared Error for each Model')

ggplot(metrics, aes(x = Model, y = Accuracy * 100)) +
  ylab('Accuracy (%)') +
  geom_bar(stat = "identity", width = 0.5) +
  ggtitle('Accuracy of each Model')

accuracy.rank = rbind(accuracy.rank.lm, accuracy.rank.interact, accuracy.rank.lasso,
                      accuracy.rank.lasso.2, accuracy.rank.tree, accuracy.rank.rf)

ggplot(accuracy.rank, aes(x = PLAYOFFS, y = accuracy * 100, fill = name)) +
  geom_bar(stat = "identity", position = "dodge") +
  ylab('Accuracy (%)') +
  xlab("Playoff Rankings") +
  ggtitle('Accuracy for each rank by Model') +
  scale_x_continuous(breaks = accuracy.rank$PLAYOFFS)
