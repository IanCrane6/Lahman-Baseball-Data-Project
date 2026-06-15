import pandas as pd
import numpy as np

fielding = pd.read_csv('lahman_1871-2025_csv/Fielding.csv')
batting = pd.read_csv('lahman_1871-2025_csv/Batting.csv')
pitching = pd.read_csv('lahman_1871-2025_csv/Pitching.csv')
salaries = pd.read_csv('lahman_1871-2025_csv/Salaries.csv')
teams = pd.read_csv('lahman_1871-2025_csv/Teams.csv')
appearances = pd.read_csv('lahman_1871-2025_csv/Appearances.csv')
allstars = pd.read_csv('lahman_1871-2025_csv/AllStarFull.csv')

teams = teams[(teams['yearID'] >= 2010) & (teams['yearID'] <= 2016)]

salary_cols = salaries.groupby(['yearID', 'teamID']).agg(Total_Salary =('salary', 'sum'),
                                                         Max_Salary = ('salary', 'max'),
                                                         Stdev_Salary=('salary', 'std')).reset_index()

appears = appearances[['yearID', 'teamID', 'playerID', 'G_p', 'G_all']].drop_duplicates(['yearID', 'teamID', 'playerID']).copy()

salary_classes = salaries.merge(appears, on=['yearID', 'teamID', 'playerID'], how='left')
salary_classes['G_p'] = salary_classes['G_p'].fillna(0)
salary_classes['G_all'] = salary_classes['G_all'].fillna(0)
salary_classes['Pitcher'] = salary_classes['G_p'] > (salary_classes['G_all'] *0.5)

pitcher_salary = salary_classes[salary_classes['Pitcher']] \
.groupby(['yearID', 'teamID'])['salary'].sum().reset_index(name = 'Pitcher_Salary')
batter_salary = salary_classes[~salary_classes['Pitcher']] \
.groupby(['yearID', 'teamID'])['salary'].sum().reset_index(name = 'Batter_Salary')

allstar_nums = allstars.groupby(['yearID', 'teamID']).size().reset_index(name = 'Allstar_Num')

teams = teams.copy()
teams['Batting_Avg'] = teams['H'] / teams['AB']



df = teams[['yearID', 'teamID', 'W', 'G', 'ERA', 'HR', 'E', 'attendance', 'Batting_Avg']].copy()
df = df.merge(salary_cols, on=['yearID', 'teamID'], how='inner')
df = df.merge(pitcher_salary, on=['yearID', 'teamID'], how='left')
df = df.merge(batter_salary, on=['yearID', 'teamID'], how='left')
df = df.merge(allstar_nums, on=['yearID', 'teamID'], how='left')
df = df.rename(columns = {'attendance': 'Attendance'})

df['Allstar_Num'] = df['Allstar_Num'].fillna(0)
df['Pitcher_Salary'] = df['Pitcher_Salary'].fillna(0)
df['Batter_Salary'] = df['Batter_Salary'].fillna(0)


for col in ['Total_Salary', 'Max_Salary', 'Stdev_Salary', 'Pitcher_Salary', 'Batter_Salary']:
    year_median = df.groupby('yearID')[col].transform('median')
    df[f'{col}_Adjusted'] = df[col] / year_median.replace(0, np.nan)

df['Stdev_Salary_Adjusted'] = df['Stdev_Salary_Adjusted'].fillna(0)

df['Pitcher_Batter_Ratio'] = df['Pitcher_Salary_Adjusted'] / (df['Batter_Salary_Adjusted'] + 1)

attendance_median = df['Attendance'].median()
df['Attendance'] = df['Attendance'] / attendance_median

df = df.dropna(subset=['W', 'ERA', 'HR', 'E', 'Attendance', 'Batting_Avg',
                       'Total_Salary_Adjusted', 'Max_Salary_Adjusted', 'Stdev_Salary_Adjusted'])

pd.set_option('display.max_columns', 500)
print(df[['W', 'ERA', 'HR', 'E', 'Attendance', 'Batting_Avg',
          'Total_Salary_Adjusted', 'Max_Salary_Adjusted', 'Stdev_Salary_Adjusted',
          'Pitcher_Salary_Adjusted', 'Batter_Salary_Adjusted']].describe())

df.to_csv('refined_feature_data.csv', index=False)



