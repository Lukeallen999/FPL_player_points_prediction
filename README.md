# Forecasting Fantasy Premier League Player Points Using Multiple Linear Regression
## Executive Summary
Fantasy Premier League (FPL) is a game in which users select a squad of Premier League players who earn points based on real match performances. This project aims to help users maximise points by enabling informed decision making through multiple regression models that predict player points using historical performance data. A combination of player, performance and fixture variables were used to estimate points in future gameweeks (GWs).

The dataset was created by combining historical data for current players with current season data, before being cleaned and split into training and testing sets. Model performance was evaluated using $R^2$, RMSE and MAE, the results suggest that linear regression can identify a moderate relationship between historical player performance and FPL points.

While the model performs well for regularly playing player, its accuracy may decrease overtime and for those with limited minutes or matches. Overall, the analysis predicts that Igor Thiago is predicted to achieve the highest number of points and Nordi Mukiele is forecasted to achieve the highest points per million.

## Project background

FPL is an online football game played by over 12.8 million users (Wikipedia, 2026) in which a user selects a squad of fifteen Premier League players (two goalkeepers, five defenders, five midfielders and three forwards) who earn points based on real match performances. Users have a budget of £100,000,000 to build their squad, with a maximum of three players allowed per club, users can transfer players between gameweeks (Fantasy Premier League, no date). Player prices are set before the season starts but change dynamically throughout the season depending on player demand (Editorial, 2026). Players earn points for actions such as minutes played, goals scored, goals assisted and keeping clean sheets however points can be deducted for conceding goals or receiving cards; points earned for actions vary per playing position (The Scout, 2025).  



























