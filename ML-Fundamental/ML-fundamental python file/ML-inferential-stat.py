# %% [markdown]
# # Inferential Statistics
# 
# - Descriptive statistics is the branch of statistics that deals with the collection, analysis, and interpretation of data from a sample or population. It involves summarizing and describing the main features of the data, such as measures of central tendency (e.g., mean, median, mode) and measures of dispersion (e.g., range, variance, standard deviation). Descriptive statistics aims to provide a clear and concise summary of the data that can be easily understood and communicated to others.
# 
# - Inferential statistics, on the other hand, is the branch of statistics that deals with drawing conclusions or making predictions about a larger population based on a sample of data. It involves using probability theory to estimate population parameters (e.g., mean, proportion) from sample statistics (e.g., sample mean, sample proportion) and testing hypotheses about the relationship between variables. Inferential statistics aims to make generalizations about the population based on the information obtained from the sample.`
# 
# ### Main uses of the Inferential statistics
# 
# Inferential statistics is used to make generalizations and predictions about populations based on sample data. Some of the main uses of inferential statistics are:
# 
# 1. **Estimation:** Inferential statistics can be used to estimate population parameters (such as the mean or proportion) based on sample data.
# 2. **Hypothesis testing:** Inferential statistics can be used to test hypotheses about population parameters using sample data.
# 3. **Prediction:** Inferential statistics can be used to make predictions about future observations or outcomes based on past data.
# 4. **Generalization:** Inferential statistics can be used to generalize findings from a sample to the larger population.
# 5. **Decision-making:** Inferential statistics can be used to support decision-making in various fields, such as business, medicine, and social sciences.
# 
# Overall, inferential statistics allows researchers to draw conclusions about populations based on sample data, which can be used to make informed decisions and predictions.
# 
# ### Descriptive versus inferential statistics
# 
# | Descriptive Statistics	| Inferential Statistics |
# |---------------------------|------------------------|
# | Describes sample data	| Makes inferences about population based on sample data |
# | Provides summary measures such as mean, median, mode, standard deviation, etc.	| Uses sample statistics to estimate population parameters |
# | Helps in data exploration and visualization	| Tests hypotheses about population parameters |
# | Useful in describing and summarizing data	| Helps in making predictions about future observations or outcomes |
# | Examples: frequency distributions, measures of central tendency, measures of dispersion, etc.	| Examples: t-tests, ANOVA, regression analysis, etc. |
# 
# ### Sampling error in inferential statistics
# 
# - Since the size of a sample is always smaller than the size of the population, some of the population isn’t captured by sample data. This creates sampling error, which is the difference between the true population values (called parameters) and the measured sample values (called statistics).
# 
# - Sampling error arises any time you use a sample, even if your sample is random and unbiased. For this reason, there is always some uncertainty in inferential statistics. However, using probability sampling methods reduces this uncertainty.
# 
# ### Estimating population parameters from sample statistics
# 
# The characteristics of samples and populations are described by numbers called statistics and parameters:
# 
# - A statistic is a measure that describes the sample (e.g., sample mean).
# - A parameter is a measure that describes the whole population (e.g., population mean).
# 
# **Sampling error** is the difference between a parameter and a corresponding statistic. Since in most cases you don’t know the real population parameter, you can use inferential statistics to estimate these parameters in a way that takes sampling error into account.
# 
# **Types of estimates:** There are two important types of estimates you can make about the population: 
# 
# 1. **Point estimates:** A point estimate is a single value estimate of a parameter. For instance, a sample mean is a point estimate of a population mean.
# 2. **Interval estimates:** An interval estimate gives you a range of values where the parameter is expected to lie. A confidence interval is the most common type of interval estimate.
# 
# Both types of estimates are important for gathering a clear idea of where a parameter is likely to lie.
# 
# ### Steps to follow to eastimate the population parameters from sample statistics
# 
# The following steps outline the process of estimating population parameters from sample statistics:
# 
# - **Define the population of interest:** Clearly define the population from which the sample is drawn.
# - **Determine the sampling technique:** Determine the sampling technique to be used to ensure that the sample is representative of the population.
# - **Collect the sample:** Collect a sample of appropriate size from the population using the chosen sampling technique.
# - **Calculate sample statistics:** Calculate the appropriate sample statistics (e.g., mean, standard deviation) based on the collected sample.
# - **Make inferences:** Use the sample statistics to make inferences about the population parameters (e.g., mean, standard deviation).
# - **Estimate population parameters:** Use the sample statistics to estimate the population parameters, including point estimates and confidence intervals.
# - **Assess the reliability of the estimate:** Determine the reliability of the estimate using statistical tests such as hypothesis testing and confidence intervals.
# 
# Overall, estimating population parameters from sample statistics involves collecting a representative sample, calculating sample statistics, and using these statistics to make inferences about the population parameters. The process requires careful consideration of the sampling technique, sample size, and reliability of the estimates.

# %% [markdown]
# ### Hypothesis testing
# 
# - Hypothesis testing is a formal process of statistical analysis using inferential statistics. The goal of hypothesis testing is to compare populations or assess relationships between variables using samples. 
# 
# - Hypotheses, or predictions, are tested using statistical tests. Statistical tests also estimate sampling errors so that valid inferences can be made.
# 
# - Statistical tests can be:
# 
#     - **Parametric**: parametric tests are considered more statistically powerful because they are more likely to detect an effect if one exists. Parametric tests make assumptions that include the following:
#         - the population that the sample comes from follows a normal distribution of scores
#         - the sample size is large enough to represent the population
#         - the variances, a measure of variability, of each group being compared are similar
#         - Example: t-test, ANOVA, Regression analysis, Pearson correlation coefficient.
#         - **t-test:** Used to compare the means of two groups when the data are normally distributed and the variances of the two groups are equal.
#         - **ANOVA:** Used to compare the means of three or more groups when the data are normally distributed and the variances of the groups are equal.
#         - **Regression analysis:** Used to model the relationship between two or more variables when the data are normally distributed and the assumptions of the model are met.
#         - **Pearson correlation coefficient:** Used to measure the strength and direction of the linear relationship between two continuous variables when the data are normally distributed.
# 
# 
#     - **Non-parametric**: When your data violates any of these assumptions, non-parametric tests are more suitable. Non-parametric tests are called “distribution-free tests” because they don’t assume anything about the distribution of the population data.
#         - Example: Wilcoxon signed-rank test, Mann-Whitney U test, Kruskal-Wallis test, Spearman correlation coefficient
#         * **Wilcoxon signed-rank test:** Used to compare the medians of two related samples when the data are not normally distributed.
#         * **Mann-Whitney U test:** Used to compare the medians of two independent groups when the data are not normally distributed.
#         * **Kruskal-Wallis test:** Used to compare the medians of three or more groups when the data are not normally distributed.
#         * **Spearman correlation coefficient:** Used to measure the strength and direction of the monotonic relationship between two continuous variables when the data are not normally distributed.

# %% [markdown]
# - Statistical tests come in three forms: 
# 1. **comparison test:** Comparison tests assess whether there are differences in means, medians or rankings of scores of two or more groups. To decide which test suits your aim, consider whether your data meets the conditions necessary for parametric tests, the number of samples, and the levels of measurement of your variables.
#     
# | Comparison test	| Parametric?	| What’s being compared?	| Samples |
# |-------------------|---------------|---------------------------|---------|
# | t-test	| Yes	| Means	| 2 samples |
# | ANOVA	| Yes	| Means	| 3+ samples |
# | Mood’s median	| No	| Medians	| 2+ samples |
# | Wilcoxon signed-rank	| No	| Distributions	| 2 samples |
# | Wilcoxon rank-sum (Mann-Whitney U)	| No	| Sums of rankings	| 2 samples |
# | Kruskal-Wallis H	| No	| Mean rankings	| 3+ samples |
# 
# 2. **correlation test:** Correlation tests determine the extent to which two variables are associated. Although Pearson’s r is the most statistically powerful test, Spearman’s r is appropriate for interval and ratio variables when the data doesn’t follow a normal distribution. The chi square test of independence is the only test that can be used with nominal variables.
# 
# | Correlation test	| Parametric?	| Variables |
# |-------------------|---------------|-----------|
# | Pearson’s r	| Yes	| Interval/ratio variables |
# | Spearman’s r	| No	| Ordinal/interval/ratio variables |
# | Chi square test of independence	| No	| Nominal/ordinal variables |
# 
# 3. **regression test:** Regression tests demonstrate whether changes in predictor variables cause changes in an outcome variable. You can decide which regression test to use based on the number and types of variables you have as predictors and outcomes. Most of the commonly used regression tests are parametric. If your data is not normally distributed, you can perform data transformations.Data transformations help you make your data normally distributed using mathematical operations, like taking the square root of each value.
# 
# | Regression test	| Predictor	| Outcome |
# |-------------------|-----------|---------|
# | Simple linear regression	| 1 interval/ratio variable	| 1 interval/ratio variable |
# | Multiple linear regression	| 2+ interval/ratio variable(s)	| 1 interval/ratio variable |
# | Logistic regression	| 1+ any variable(s)	| 1 binary variable |
# | Nominal regression	| 1+ any variable(s)	| 1 nominal variable |
# | Ordinal regression	| 1+ any variable(s)	| 1 ordinal variable |

# %% [markdown]
# > **Degrees of Freedom:** Degrees of freedom, often represented by v or df, is the number of independent pieces of information used to calculate a statistic. It’s calculated as the sample size minus the number of restrictions.

# %% [markdown]
# ### Step-by-step guide to hypothesis testing
# 
# Here's a step-by-step guide to hypothesis testing:
# 
# 1. **Formulate the null and alternative hypotheses**: 
# 
#     The null hypothesis (denoted $H_0$) is the hypothesis that there is no significant difference between two or more groups, or no significant relationship between two or more variables. The alternative hypothesis (denoted $H_a$) is the hypothesis that there is a significant difference or relationship. 
#     - For example, in a clinical trial, the null hypothesis might be that there is no difference in effectiveness between a new drug and a placebo, while the alternative hypothesis might be that the new drug is more effective than the placebo. 
#     - <a href="#null-and-alternative-hypothesis" style="display:inline-block;padding:5px 10px;background-color:#337ab7;color:#fff;text-decoration:none;border-radius:5px;font-size:14px;">Go to detailed section</a>
# 
# 2. **Choose a significance level:** The significance level (denoted $\alpha$) is the probability of rejecting the null hypothesis when it is actually true. The most common significance level used in hypothesis testing is 0.05 (or 5%). 
#     - <a href="#significance-level" style="display:inline-block;padding:5px 10px;background-color:#337ab7;color:#fff;text-decoration:none;border-radius:5px;font-size:14px;">Go to detailed section</a>
# 
# 3. **Select an appropriate statistical test:** The choice of statistical test depends on the type of data and the research question. For example, a t-test might be used to compare the means of two groups, while ANOVA might be used to compare the means of three or more groups.
#     - <a href="#selecting-a-appropriate-statistical-test" style="display:inline-block;padding:5px 10px;background-color:#337ab7;color:#fff;text-decoration:none;border-radius:5px;font-size:14px;">Go to detailed section</a>
# 
# 4. **Calculate the test statistic:** In hypothesis testing, the test statistic is a numerical value calculated from the sample data that is used to assess the evidence against the null hypothesis. The test statistic measures how far the sample estimate is from the null hypothesis value in terms of the standard error of the estimate. The choice of test statistic depends on the type of data and the assumptions of the statistical test. The formula for the test statistic depends on the specific test being used. 
# 
#     - For example, in a t-test, the test statistic is calculated as: 
#     
#         $$t = \frac{(\bar{x} - \mu)}{s/\sqrt{n}}$$
#         where $x̄$ is the sample means, $\mu$ is the hypothesized population mean, $s$ is the sample standard deviation, and $n$ is the sample size. The t-test produces a t-value, which is compared to a critical value from the t-distribution based on the degrees of freedom and the desired significance level.
#     - The z-test, on the other hand, is used when the population standard deviation is known or when the sample size is large (typically $n\geq 30$). The formula for the z-test is:
#     
#         $$z = \frac{x̄ - μ}{\sigma/\sqrt{n}}$$
#         
#         where $\bar{x}$ is the sample mean, $\mu$ is the hypothesized population mean, $\sigma$ is the population standard deviation, and $n$ is the sample size. The z-test produces a z-value, which is compared to a critical value from the standard normal distribution based on the desired significance level.
# 
#     Both the t-test and z-test are used to test hypotheses about the population mean based on sample data. The choice between the two tests depends on the nature of the data and the sample size. If the population standard deviation is unknown and the sample size is small, the t-test is appropriate. If the population standard deviation is known or the sample size is large, the z-test is appropriate.
# 
# 5. **Calculate the p-value:** The p-value is the probability of obtaining a test statistic as extreme as, or more extreme than, the observed test statistic, assuming the null hypothesis is true. The p-value is calculated based on the distribution of the test statistic under the null hypothesis. If the p-value is less than the significance level (α), the null hypothesis is rejected in favor of the alternative hypothesis.
#     - To calculate the p-value, we need to compare the test statistic to its sampling distribution under the null hypothesis. "_The p-value is the area under the sampling distribution curve that is more extreme than the observed test statistic._" 
#         - If the p-value is small (typically less than the significance level), it indicates that the observed data is unlikely to have occurred by chance alone, and the null hypothesis should be rejected in favor of the alternative hypothesis.
# 
#     - The exact calculation of the p-value depends on the type of test and the distribution of the test statistic under the null hypothesis. For example, 
#         - in a t-test, the p-value can be calculated using the t-distribution with n-1 degrees of freedom, where n is the sample size. - In a z-test, the p-value can be calculated using the standard normal distribution. 
#         - In other tests, such as the chi-square test or F-test, the p-value is calculated using the chi-square or F distribution, respectively.
# 
#     - In general, the smaller the p-value, the stronger the evidence against the null hypothesis. 
#         - A p-value less than the significance level (usually 0.05 or 0.01) indicates that the result is statistically significant and that the null hypothesis should be rejected. 
#         - A p-value greater than the significance level indicates that the result is not statistically significant, and the null hypothesis should not be rejected.
# 
#     - In summary, the p-value measures the strength of evidence against the null hypothesis and is calculated by comparing the test statistic to its sampling distribution under the null hypothesis. The exact calculation of the p-value depends on the type of test and the distribution of the test statistic under the null hypothesis. A small p-value indicates that the result is statistically significant and that the null hypothesis should be rejected.
# 
# 6. **Interpret the results:** If the p-value is less than the significance level, the result is considered statistically significant, and the null hypothesis is rejected. If the p-value is greater than the significance level, the result is not considered statistically significant, and the null hypothesis cannot be rejected.

# %% [markdown]
# **Example:** To illustrate this process, let's consider an example: 
# 
# Suppose we want to test whether there is a difference in the average height of men and women. We collect a sample of 100 men and 100 women, and measure their heights. We want to test the null hypothesis that there is no difference in height between men and women, against the alternative hypothesis that there is a difference.
# 
# - *Null hypothesis*: $H_0$: $\mu_1 = \mu_2$ (there is no difference in height between men and women). 
# 
# - *Alternative hypothesis:* $H_a$: $\mu_1 \neq \mu_2$ (there is a difference in height between men and women)
# 
# - *Significance level*: $\alpha = 0.05$
# 
# - *Statistical test*: We can use a two-sample t-test to compare the means of the two groups.
# 
# - *Test statistic*: The test statistic for a two-sample t-test is calculated as:
# 
#     t = (x̄1 - x̄2) / (s / sqrt(n1 + n2))
# 
#     where x̄1 and x̄2 are the sample means, s is the pooled standard deviation, and n1 and n2 are the sample sizes. In our example, let's assume that the sample mean height for men is 175 cm and the sample mean height for women is 162 cm. The pooled standard deviation (s) is calculated as:
# 
#     $s = \sqrt{\frac{(n1 - 1) * s1^2 + (n2 - 1) * s2^2}{(n1 + n2 - 2)}}$.
# 
#     where s1 and s2 are the sample standard deviations for the men and women, respectively. Let's assume that s1 = 6 cm and s2 = 5 cm. Then:
# 
#     s = 5.524 cm
# 
#     Plugging in the values, we get:
# 
#     $t = \frac{175 - 162}{{5.524 /\sqrt{100 + 100}}} = 12.215$
# 
# - *P-value*: The p-value is the probability of getting a t-statistic as extreme as 12.215, assuming that there is no difference in height between men and women. The p-value can be calculated using a t-distribution with 198 degrees of freedom (the sum of the sample sizes minus 2). Using a t-table or a statistical software, we find that the p-value is much less than 0.05 (in fact, it is practically 0), indicating strong evidence against the null hypothesis.
# 
# - *Interpretation*: Since the p-value is less than the significance level, we reject the null hypothesis and conclude that there is a statistically significant difference in height between men and women.
# 
# This example illustrates how hypothesis testing can be used to draw conclusions about differences between groups based on sample data. The choice of statistical test, calculation of the test statistic, and interpretation of the results can vary depending on the specific research question and data being analyzed.

# %% [markdown]
# ## [Null and Alternative hypothesis](#null-and-alternative-hypothesis)
# 
# Formulating the null and alternative hypotheses is a critical step in hypothesis testing.
# 
# 1. **Null hypothesis ($H_0$)**: The null hypothesis ($H_0$) is the statement that there is no significant difference or relationship between the variables being studied. 
# 2. **Alternative hypothesis ($H_a$ or $H_1$)** is the statement that there is a significant difference or relationship between the variables.
# 
# - In statistical testing, the null hypothesis is assumed to be true until there is evidence to the contrary. The alternative hypothesis represents the researcher's hypothesis or the hypothesis they are trying to prove.
# 
# - The null and alternative hypotheses must be mutually exclusive and collectively exhaustive. That means that they cover all possible outcomes and that there is no overlap between them. In other words, one of the hypotheses must be true, but not both.
# 
# - **Example:** suppose a researcher is interested in whether a new drug is effective in reducing blood pressure. They could formulate the null and alternative hypotheses as follows:
# 
#     $H_0$: The new drug has no effect on blood pressure.
#         
#     $H_a$: The new drug has a significant effect on blood pressure.
# 
#     Alternatively, suppose a researcher is interested in whether there is a difference in job satisfaction between men and women. They could formulate the null and alternative hypotheses as follows:
# 
#     $H_0$: There is no significant difference in job satisfaction between men and women.
# 
#     $H_a$: There is a significant difference in job satisfaction between men and women.
# 
#     The null and alternative hypotheses can be one-tailed or two-tailed, depending on the direction of the expected difference or relationship between the variables. A one-tailed hypothesis predicts the direction of the effect (e.g., the new drug will lower blood pressure), while a two-tailed hypothesis does not predict the direction of the effect (e.g., there is a difference in job satisfaction between men and women).
# 
# In summary, formulating the null and alternative hypotheses is a critical step in hypothesis testing, as it defines the research question and the direction of the analysis. The hypotheses must be mutually exclusive and collectively exhaustive, and their formulation depends on the research question and the expected relationship or difference between the variables being studied.

# %% [markdown]
# ## [Significance level ($\alpha$)](#significance-level)
# 
# Choosing a significance level is another important step in hypothesis testing. The significance level, denoted by $\alpha$, is the probability of rejecting the null hypothesis when it is actually true. In other words, it is the probability of making a Type I error.
# 
# - A Type I error occurs when we reject the null hypothesis when it is actually true. This is also known as a false positive. The significance level is therefore a measure of how willing we are to make a Type I error.
# 
# - The most commonly used significance level is 0.05, which means that we are willing to accept a 5% chance of making a Type I error. However, the significance level can be set to any value, depending on the researcher's preferences, the consequences of making a Type I error, and the strength of the evidence needed to reject the null hypothesis.
# 
# - It is important to note that the choice of significance level is not independent of the sample size and the statistical power of the test. A smaller sample size or a weaker effect size may require a lower significance level to avoid Type I errors. On the other hand, a larger sample size or a stronger effect size may allow for a higher significance level.
# 
# - To choose a significance level, researchers should consider the following factors:
# 
#     1. The consequences of making a Type I error: If the consequences of rejecting the null hypothesis when it is actually true are severe, a lower significance level should be chosen.
# 
#     2. The strength of the evidence needed to reject the null hypothesis: If strong evidence is required to reject the null hypothesis, a lower significance level should be chosen.
# 
#     3. The sample size and effect size: If the sample size is small or the effect size is weak, a lower significance level may be needed to avoid Type I errors.
# 
# In summary, choosing a significance level is a critical step in hypothesis testing that involves balancing the risk of Type I errors with the strength of evidence needed to reject the null hypothesis. The significance level is usually set to 0.05 but can be adjusted based on the research question, the consequences of making an error, and the characteristics of the data being analyzed.

# %% [markdown]
# ## [Selecting a appropriate statistical test](#selecting-a-appropriate-statistical-test)
# 
# Selecting an appropriate statistical test is a crucial step in hypothesis testing. The choice of test depends on the nature of the research question, the type of data being analyzed, and the assumptions of the statistical test. Selecting the wrong test can lead to incorrect conclusions and can invalidate the results of the analysis.
# 
# There are many different statistical tests available, and choosing the right one can be challenging. However, the following factors can guide the selection of an appropriate test:
# 
# 1. **Type of data:** The type of data being analyzed (e.g., continuous, categorical, ordinal) can determine the appropriate statistical test. For example, t-tests and ANOVA are appropriate for continuous data, while chi-squared tests are appropriate for categorical data.
# 
# 2. **Sample size:** The sample size can also influence the choice of test. For small sample sizes, nonparametric tests may be more appropriate, while for larger sample sizes, parametric tests may be more powerful.
# 
# 3. **Number of groups:** The number of groups being compared can also influence the choice of test. For example, t-tests are appropriate for comparing two groups, while ANOVA is appropriate for comparing three or more groups.
# 
# 4. **Assumptions of the test:** Most statistical tests have certain assumptions that must be met in order for the test to be valid. For example, t-tests assume that the data are normally distributed, while ANOVA assumes that the variances of the groups are equal. Violating these assumptions can lead to incorrect conclusions.
# 
# 5. **Research question:** Finally, the research question itself can guide the choice of test. For example, if the research question involves the relationship between two variables, a correlation or regression analysis may be appropriate.
# 
# In summary, selecting an appropriate statistical test involves considering the type of data being analyzed, the sample size, the number of groups, the assumptions of the test, and the research question. By carefully selecting the appropriate test, researchers can ensure that their results are valid and can make meaningful conclusions from their data.

# %% [markdown]
# # Various kinds of tests
# 
# Comparison test
# 1. t-test	
# 2. ANOVA	
# 3. Mood’s median	
# 4. Wilcoxon signed-rank
# 5. Wilcoxon rank-sum (Mann-Whitney U)	
# 6. Kruskal-Wallis H	
# 
# Correlation test
# 1. Pearson’s r	
# 2. Spearman’s r	
# 3. Chi square test of independence
# 
# Regression test
# 1. Simple linear regression	
# 2. Multiple linear regression	
# 3. Logistic regression	
# 4. Nominal regression	
# 5. Ordinal regression

# %% [markdown]
# ## 1. t-test
# 
# The t-test is a statistical test used to determine if there is a significant difference between the means of two groups. There are two main types of t-tests: 
# 1. the independent samples t-test and 
# 2. the paired samples t-test.
# 
# Independent Samples t-Test:
# The independent samples t-test is used to compare the means of two independent groups. The null hypothesis is that the two groups have equal means, while the alternative hypothesis is that they have different means.
# 
# The formula for the independent samples t-test is:
# 
# t = (x̄1 - x̄2) / (s√(1/n1 + 1/n2))
# 
# Where:
# x̄1 and x̄2 are the means of the two groups
# s is the pooled standard deviation
# n1 and n2 are the sample sizes of the two groups
# 
# The independent samples t-test assumes that the variances of the two groups are equal. If they are not, a modified version of the t-test called Welch's t-test can be used.
# 
# Paired Samples t-Test:
# The paired samples t-test is used to compare the means of two related groups, such as before and after measurements of the same group. The null hypothesis is that the mean difference between the two groups is zero, while the alternative hypothesis is that the mean difference is different from zero.
# 
# The formula for the paired samples t-test is:
# 
# t = d̄ / (s/√n)
# 
# Where:
# d̄ is the mean difference between the two groups
# s is the standard deviation of the differences
# n is the sample size
# 
# The paired samples t-test assumes that the differences between the two groups are normally distributed.
# 
# Interpretation:
# After calculating the t-value and degrees of freedom, the p-value can be obtained from the t-distribution table. If the p-value is less than the significance level, then the null hypothesis is rejected, indicating that there is a significant difference between the means of the two groups. If the p-value is greater than the significance level, then the null hypothesis cannot be rejected, indicating that there is not a significant difference between the means of the two groups.

# %% [markdown]
# ## 2. ANOVA (Analysis of Variance) test
# 
# ANOVA (Analysis of Variance) is a statistical test used to determine whether there are significant differences between the means of three or more groups. It tests the null hypothesis that all group means are equal, against the alternative hypothesis that at least one group mean is different from the others.
# 
# The ANOVA test is based on the F-statistic, which is calculated by dividing the variance between the groups by the variance within the groups. The F-statistic follows an F-distribution with degrees of freedom based on the number of groups and the number of observations.
# 
# Here are the steps to perform an ANOVA test:
# 
# 1. Formulate the null and alternative hypotheses:
#     - H0: μ1 = μ2 = μ3 = ... = μk (all group means are equal)
#     - Ha: at least one group mean is different from the others
# 
# 2. Choose a significance level (α) and determine the degrees of freedom for the F-distribution:
#     - Degrees of freedom between groups: k - 1
#     - Degrees of freedom within groups: N - k, where N is the total number of observations
# 
# 3. Collect and organize the data into groups.
# 
# 4. Calculate the sum of squares between groups (SSbetween):
#     - SSbetween = ∑ni(x̄i - x̄)^2, 
#     
#     where ni is the number of observations in group i, x̄i is the mean of group i, and x̄ is the overall mean
# 
# 5. Calculate the sum of squares within groups (SSwithin):
#     - SSwithin = ∑(xi - x̄i)^2, 
#     
#     where xi is the ith observation in group i
# 
# 6. Calculate the mean square between groups (MSbetween):
#     - MSbetween = SSbetween / (k - 1)
# 
# 7. Calculate the mean square within groups (MSwithin):
#     - MSwithin = SSwithin / (N - k)
# 
# 8. Calculate the F-statistic:
#     - F = MSbetween / MSwithin
# 
# 9. Calculate the p-value associated with the F-statistic using a table of F-distributions or a statistical software package.
# 
# 10. Compare the p-value to the significance level. If the p-value is less than the significance level, reject the null hypothesis and conclude that there is a significant difference between the means of at least two groups. If the p-value is greater than the significance level, fail to reject the null hypothesis and conclude that there is not enough evidence to suggest a significant difference between the means of any of the groups.
# 
# **Example:** Here's an example of an ANOVA test:
# 
# Suppose we want to compare the average test scores of students in three different schools (A, B, and C). We randomly sample 10 students from each school and obtain the following data:
# 
# - **School A:** 75, 80, 82, 85, 88, 90, 92, 94, 95, 98
# - **School B:** 70, 75, 77, 80, 83, 85, 87, 88, 90, 92
# - **School C:** 65, 70, 72, 75, 78, 80, 82, 84, 85, 88
# 
# 1. Formulate the null and alternative hypotheses:
#     - H0: μA = μB = μC
#     - Ha: at least one group mean is different from the others
# 
# 2. Choose a significance level (α) and determine the degrees of freedom for the F-distribution:
#     - α = 0.05
#     - Degrees of freedom between groups: 3 - 1 = 2
#     - Degrees of freedom within groups: 30 - 3 = 27
# 
# 3. Collect and organize the data into groups.
# 
# 4. Calculate the sum of squares between groups (SSbetween):
#     - SSbetween

# %%



