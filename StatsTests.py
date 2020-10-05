import pandas as pd
import scipy.stats as stats

##@file StatsTests.py
## Documentation for StatsTest.py
#
# we would be using t-tests and ANOVA for analysis. Script.stat package is used; ttest_ind and f_oneway functions are performed.
# Null Hypothesis (H0): There is no significant difference between the strategies. Alternative Hypothesis (H1): There are significant differences between the strategies. We use 0.05 for the significant level.
df= pd.read_csv('data/MeanRewardsResult.csv')
t_test = stats.ttest_ind(df['EGreedy'], df['EDecreasing'])
ANOVA1= stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid1'])
ANOVA2 =stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid2'])
ANOVA3 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid3'])
ANOVA4 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid4'])
ANOVA5 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid5'])
print("T-Test of Epsilon-Greedy and Epsilon-Decreasing:", t_test, "and we fail to reject the null hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#1:", ANOVA1, "and we fail to reject the null hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#2:", ANOVA2, "and we would like to reject the null hypothesis in favor of the alternative hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#3:", ANOVA3, "and we fail to reject the null hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#4:", ANOVA4, "and we would like to reject the null hypothesis in favor of the alternative hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#5:", ANOVA5, "and we fail to reject the null hypothesis.")
