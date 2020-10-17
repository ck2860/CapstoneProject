import pandas as pd
import scipy.stats as stats
from Plot import *
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore")

## @package StatsTest
## Documentation for StatsTest.py
# we would be using t-tests and ANOVA for analysis. Script.stat package is used; ttest_ind and f_oneway functions are performed.
# we use a Tukey post hoc analysis to confirm where the differences occurred between greedy-based strategies: Epsilon-Greedy, Epsilon-Decreasing, Hybrid#2, and Hybrid#4.

# Null Hypothesis (H0): There is no significant difference between the strategies.
# Alternative Hypothesis (H1): There are significant differences between the strategies.
# We use 0.05 for the significant level.

df = pd.read_csv('data/MeanRewardsResult.csv')  # importing data

# Performing t-tests and ANOVA
t_test = stats.ttest_ind(df['EGreedy'], df['EDecreasing'])  # T-Test on epsilon-greedy and epsilon-decreasing
ANOVA1 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid1'])  # ANOVA on epsilon-greedy, epsilon-decreasing, Hybrid#1
ANOVA2 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid2'])  # ANOVA on epsilon-greedy, epsilon-decreasing, Hybrid#2
ANOVA3 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid3'])  # ANOVA on epsilon-greedy, epsilon-decreasing, Hybrid#3
ANOVA4 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid4'])  # ANOVA on epsilon-greedy, epsilon-decreasing, Hybrid#4
ANOVA5 = stats.f_oneway(df['EGreedy'], df['EDecreasing'], df['Hybrid5'])  # ANOVA on epsilon-greedy, epsilon-decreasing, Hybrid#5

# Printing the results
print("T-Test of Epsilon-Greedy and Epsilon-Decreasing: the p-value is ", t_test[1], "and we fail to reject the null hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#1: the p-value is ", ANOVA1[1], "and we fail to reject the null hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#2: the p-value is ", ANOVA2[1], "and we would like to reject the null hypothesis in favor of the alternative hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#3: the p-value is ", ANOVA3[1], "and we fail to reject the null hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#4: the p-value is ", ANOVA4[1], "and we would like to reject the null hypothesis in favor of the alternative hypothesis.")
print("ANOVA of Epsilon-Greedy, Epsilon-Decreasing, and Hybrid#5: the p-value is ", ANOVA5[1], "and we fail to reject the null hypothesis.")

# Statsmodels.stats.multicomp package is used; only pairwise_tukeyhsd function is performed.
df1 = pd.read_csv('data/TukeyData.csv')  # importing a data set
tukey_result = pairwise_tukeyhsd(endog=df1['MeanRewards'], groups=df1['Strategy'], alpha=0.05)  # performing Tukey post hoc test
tukey_plot(tukey_result)  # calling the tukey_plot function
