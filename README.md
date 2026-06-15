# Lahman-Baseball-Data-Project
Data ETL pipeline and regression models utilizing the Lahman Baseball database. Identifying financial features and other aspects of MLB team data that impact total season wins. You can run the Lahmna_Table_Build.py file to build a refined csv file with feature data. The Wins_Model.py can be ran on the csv file that is output from the first file.

Abstract:

In baseball and more specifically Major League Baseball, the ability to accurately predict
outcomes is imperative in the statistics heavy and driven sporting event. The problem that Team 100 has
committed to solving is developing a model and visualization for key predictors that are pivotal in
determining the win rate of an MLB team. This outcome is important because total wins in a season is a
key focus for stakeholders in the sport. Total wins are a direct measure of the overall success of a team.
More wins increase the likelihood of making the postseason and ultimately playing in and winning the
World Series. Front offices and analysts care about understanding which factors truly drive wins and how
roster construction translates into results. Fans and media care about reliable win projections and clear
explanations of why teams succeed. Bettors care about identifying mispriced win totals and uncovering
statistical edges before sportsbooks adjust. Most prior work focuses on predicting individual game
outcomes rather than full season wins. Even strong models achieve only about 60% accuracy at the game
level, highlighting how difficult baseball is to predict (Soto Valero, 2016). Many studies also rely on
limited time periods, older data, or a narrow set of features, which may not reflect modern MLB dynamics
(Richards & Guell, 1998; Barry & Hartigan, 1993). Instead of predicting games one at a time, we
directly model season-level win totals.
