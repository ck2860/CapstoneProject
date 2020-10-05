import pandas as pd
import scipy.stats as stats

##@package StatsTest
#Documentation for StatsTest.py
#
# we would be using t-tests and ANOVA for analysis.No classes are created here. Script.stat package is used; ttest_ind and f_oneway functions are performed.
df= pd.read_csv('data/MeanRewardsResult.csv')
t_test = stats.ttest_ind(df['EGreedy'], df['EDecreasing'])
ANOVA1= stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid1'])
ANOVA2 =stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid2'])
ANOVA3 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid3'])
ANOVA4 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid4'])
ANOVA5 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid5'])
print("T-Test of Epsilon-Greedy and Epsilon-Decreasing:", t_test)
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#1:", ANOVA1)
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#2:", ANOVA2)
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#3:", ANOVA3)
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#4:", ANOVA4)
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#5:", ANOVA5)
