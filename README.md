# NBAPlayoffsPred
Project for Data Science for Business at Duke Fuqua School of Business.

Use NBA Regular Season Data (2020-2025) to train and build models to predict playoff performance.

# Data Preparation and Sources

Data Source - https://www.nbastuffer.com/

After manually scraping the data from this website for players and teams, we
modified columns such as Team and Rank to be consistent with the formatting across
different seasons, and also for the player data and team data. As part of the data preparation
and with the objective of training our model, we attributed categorized variables to each team
from a range of 0 to 6, within our team dataset. The categorized variables represent a past
team’s performance for each season, in the playoffs, and are used in order to best predict
future results. This is the feature variable we will aim to train our models on and use for
prediction for the 2024-2025 season. The feature variable’s breakdown is as follows:

● 0 = The team did not make the play-ins or playoffs

● 1 = The team made the play-ins but did not make the playoffs

● 2 = The team made it to the first round of playoffs

● 3 = The team made it to the conference semi-finals

● 4 = The team made it to conference finals

● 5 = The team made it to NBA finals

● 6 = The NBA champions

Moreover, as part of our data cleaning process within the players’ dataset, we are
looking at removing any player that may not have a significant impact on a team’s
performance. Lastly, we aggregated the
player data by calculating weighted averages across each season for each team to get
important metrics such as Average Offensive Rating (Avg_Ortg) and Average Turnover %
(Avg_TO) that we merge with the team data to potentially use in our modelling approach.\

# Models

Linear Regression: We use a multiple linear regression model as well as linear regression
with interaction variables. After testing a variety of different combinations, we settled on a
simple model that uses some of the aggregated player metrics as well as some team statistics
for multiple linear regression. Similarly, for linear regression with interaction variables, we
decided to use fewer variables that might have a lot of effect on rankings but include
interactions amongst them to see how it affects playoff standings.

LASSO (Least Absolute Shrinkage and Selection Operator): After testing a bunch of
different combinations of features, we find that the LASSO model is most effective when
provided many features with interactions amongst all of them and it can decide which ones to
use and which ones to not. We make predictions with minimum lambda as well as 1SE
lambda.

CART: We use a Regression Tree to identify useful features and split points to make
predictions. We run cross validation to identify the best size for the tree. Unfortunately, due to
the heavy impact certain variables such as eDIFF (Difference in offensive and defensive
efficiency) have, the tree just decides to split across the 1 variable.

Random Forest: We run a collection of regression trees using the Random Forest algorithm
(500 trees) and with a mtry parameter value of 7. We decided on this mtry value after testing
multiple values and settling on the one that got us the least RMSE.

# Evaluation + Prediction

We evaluate all models using the out of sample Root Mean Squared Error and Accuracy. We
calculate accuracy for models in 2 different ways. Firstly, we simply compute for all
predictions, how many the model was able to get right and then we compute accuracy within
each playoff ranking to see what the model struggles with predicting. RMSE tends to range
from 0.79 to 0.99, indicating that predictions are usually within 1 round of the actual playoff
rankings. Accuracy values tend to hover in the 60–66.7% range, with a low of 56.67% and a
high of 70% for Regression Tree and Lasso with minimum Lambda respectively. Accuracy
across ranks differ quite a bit, with models being able to predict teams who did not make
playoffs as well as the playoff championship quite well. Play-in teams and teams that went
out in 1st round (Ranking = 1 or 2) tend to still have decent accuracy across most models, but
accuracy tends to get worse as playoff rankings increase until the championship (Ranking =
6) with none of the models correctly predicting the runner-ups (Ranking = 5). Overall, Linear
Regression with interaction terms, Random Forest and Lasso Regression with the minimum
Lambda value perform the best, balancing a low RMSE score and a decent accuracy score.

# Deployment

The model could be continuously refreshed throughout the season, with the final
model updated after all of the data from the regular season is accounted for. This model is
intended to be used as a decision support tool to better budget marketing resources, more
efficiently train, and to start planning current contracts earlier in preparation for the NBA
Draft.

Marketing: Ranking predictions can help marketing teams allocate promotional
budgets more effectively. Teams projected to perform well can increase investment in
playoff-related campaigns, ticket pricing strategies, and merchandise pushes, while
lower-ranked teams can focus on fan engagement and community-building efforts. The
Marketing focus should be around the possibility of winning and qualifying for the play-offs,
fueling excitement to grow a larger fan base.

Training: Coaches can use performance indicators from the model to identify weak
areas to train more on, and adjust accordingly. Teams should use the model in support of their
current training regime.

Contracts: Teams can use these insights to plan negotiations and budget accordingly.
This will help with forecasting salary demands and help with the draft budget.
