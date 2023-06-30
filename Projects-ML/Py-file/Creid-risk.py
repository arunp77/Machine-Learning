# %% [markdown]
# # Role of Data Science in Risk analysis
# 
# - Risk management is an integral part of any financial institution. All businesses face a variety of risks and the risk management practice works towards maximizing the businesses’ return on investment and reducing their losses. 
# 
# <img src="project-Image/risk-2.png" width="400" height="400" />

# %% [markdown]
# ## Risk Analysis
# Risk analysis is a multi-step process aimed at mitigating the impact of risks on business operations.
#  
# Here we will discuss three points:
# 
# 1. components, 
# 2. types, 
# 3. methods
# 4. examples and 
# 5. steps on how to perform risk analysis

# %% [markdown]
# ### Difference Between Risk Assessment and Risk Analysis
# 
# - Risk assessment is just one component of risk analysis. 
# - The other components of risk analysis are risk management and risk communication. 
# - Risk management is the proactive control and evaluation of risks while risk communication is the exchange of information involving risks. 
# - Unlike risk analysis, risk assessment is primarily focused on safety and hazard identification.

# %% [markdown]
# ### Types of risk analysis
# 
# There are several types of risk analysis commonly used in various industries and sectors. Here are some of the key types of risk analysis:
# 
# - **Quantitative Risk Analysis:** This type of risk analysis involves the use of numerical data and statistical techniques to assess the likelihood and potential impact of risks. It includes methods such as probability analysis, sensitivity analysis, scenario analysis, and Monte Carlo simulations to quantify and model risks.
# 
# - **Qualitative Risk Analysis:** Qualitative risk analysis focuses on assessing risks based on subjective judgments and expert opinions rather than numerical data. It involves techniques such as risk ranking, risk categorization, risk assessments based on likelihood and impact scales, and risk matrix analysis. Qualitative risk analysis provides a qualitative understanding of risks and helps prioritize them based on their severity.
# 
# - **<span style="color:blue">Financial Risk Analysis</span>:** 
#   Financial risk analysis is specific to the evaluation and management of financial risks faced by organizations. It includes 
#   - assessing risks related to market fluctuations, 
#   - credit default, 
#   - liquidity, 
#   - interest rates, 
#   - foreign exchange, and 
#   - other financial factors. 
#   
#   Financial risk analysis often involves quantitative methods to estimate potential losses, calculate risk measures (such as Value at Risk), and develop risk mitigation strategies.
# 
#     <img src="project-Image/risk-1.png" width="400" height="400" />
# 
# - **Operational Risk Analysis:** Operational risk analysis focuses on identifying and assessing risks arising from internal processes, systems, and human factors within an organization. It involves analyzing risks associated with errors, process failures, supply chain disruptions, technology failures, compliance breaches, and other operational vulnerabilities. Operational risk analysis helps organizations understand and mitigate risks to ensure operational continuity and efficiency.
# 
# - **Strategic Risk Analysis:** Strategic risk analysis involves assessing risks associated with an organization's strategic objectives, decisions, and market dynamics. It focuses on identifying risks that could impact the achievement of strategic goals, such as changes in market trends, competitive landscape, regulatory environment, or technological advancements. Strategic risk analysis helps organizations make informed decisions and develop risk-informed strategic plans.
# 
# - **Project Risk Analysis:** Project risk analysis is specific to evaluating risks associated with individual projects. It involves identifying and assessing risks related to project scope, timelines, resources, budgets, and stakeholder expectations. Techniques such as risk identification workshops, risk registers, risk impact and probability assessments, and risk response planning are commonly used in project risk analysis.
# 
# - **Compliance Risk Analysis:** Compliance risk analysis involves assessing risks associated with non-compliance with laws, regulations, and industry standards. It focuses on evaluating risks related to legal and regulatory requirements, ethical considerations, data privacy, information security, and anti-money laundering measures. Compliance risk analysis helps organizations ensure adherence to legal and regulatory obligations and avoid penalties or reputational damage.

# %% [markdown]
# ### Financial risk analysis
# 
# #### 1. Risk assessment
# 
# Risk assessment in the financial sector refers to the process of evaluating and analyzing the potential risks that can affect financial institutions, investments, or financial transactions. It involves identifying, measuring, and mitigating risks to protect the interests of the institution, its clients, and stakeholders.
# 
# Here's a brief overview of the key aspects of risk assessment in the financial sector:
# 
# ##### 1.1. Identification of Risks 
# 
# The first step is to identify various risks that can impact the financial institution or investment, such as: 
# 
#   - **credit risk:** Credit risk refers to the potential for financial losses resulting from the failure of a borrower or counterparty to fulfill their financial obligations. It arises when borrowers or counterparties are unable to repay their loans or meet their contractual obligations. This risk can be mitigated through credit assessments, collateral requirements, diversification of credit exposures, and the use of credit derivatives. 
# 
#     **Example:** A bank lending money to individuals or businesses faces credit risk. If a borrower defaults on their loan payments, the bank may suffer financial losses.
# 
#   - **market risk:** Market risk arises from adverse changes in market conditions, such as fluctuations in stock prices, interest rates, foreign exchange rates, or commodity prices. It can lead to losses in the value of investments or portfolios. Market risk can be managed through diversification, hedging, and risk measurement techniques like VaR (Value at Risk) and stress testing.
# 
#     **Example:** An investment fund holding a portfolio of stocks is exposed to market risk. If the stock prices decline due to market downturns, the fund's value may decrease.
# 
#   - **liquidity risk:** Liquidity risk refers to the potential difficulty of buying or selling an investment quickly and at a fair price without causing significant price changes. It arises from insufficient market liquidity or an inability to convert assets into cash when needed. Liquidity risk can be managed by maintaining adequate cash reserves, diversifying funding sources, and establishing contingency funding plans.
# 
#       **Example:** A mutual fund holding illiquid assets, such as real estate or private equity, may face liquidity risk if investors want to redeem their shares, but the fund struggles to sell the underlying assets quickly.
# 
#   - **Operational risk:** Operational risk is the potential for losses resulting from inadequate or failed internal processes, systems, human errors, or external events. It encompasses risks related to technology, fraud, legal compliance, and business continuity. Operational risk can be mitigated through proper internal controls, staff training, disaster recovery plans, and risk monitoring.
# 
#       **Example:** A cyber-attack on a financial institution's systems that compromises customer data and disrupts operations represents operational risk. 
# 
#   - **Regulatory Risk:** Regulatory risk arises from changes in laws, regulations, or government policies that impact the financial industry. It includes the risk of non-compliance with applicable regulations, which can lead to financial penalties, reputational damage, or restrictions on business activities. Regulatory risk can be managed through robust compliance programs, staying updated on regulatory changes, and engaging with regulatory authorities.
# 
#       **Example:** A bank faces regulatory risk if new legislation imposes stricter capital requirements, necessitating adjustments to its operations and capital structure.
# 
#   - **Reputational Risk:** Reputational risk refers to the potential loss of reputation or public trust in an organization due to negative perceptions or events. It arises from actions, behaviors, or incidents that damage the public image or brand value. Reputational risk can be mitigated by maintaining high ethical standards, providing quality products/services, effective crisis management, and transparent communication with stakeholders.
# 
#       **Example:** A scandal involving unethical practices in a financial institution can result in reputational risk, leading to customer loss, decreased investor confidence, and legal consequences.
# 
#     Understanding and managing these various types of risks is crucial for financial institutions and investors to protect their interests, ensure regulatory compliance, and maintain financial stability. Risk management practices, such as diversification, risk measurement, and mitigation strategies, play a significant role in minimizing the impact of these risks.
# 
# ##### 1.2. Risk Measurement
# 
# Once risks are identified, they need to be quantified and measured to assess their potential impact. This involves analyzing historical data, using statistical models, and evaluating market conditions to estimate the likelihood and severity of risks.
# 
# ##### 1.3. Risk Mitigation
# 
# After measuring risks, financial institutions implement strategies to mitigate and manage them. This may include:
#   - setting risk tolerance levels,
#   - implementing risk control measures, 
#   - diversifying investments, 
#   - hedging strategies, or 
#   - establishing contingency plans.
# 
# ##### 1.4. Risk Monitoring
# 
# Ongoing monitoring is crucial to ensure that risks are continually assessed and managed effectively. Regular monitoring helps identify emerging risks, evaluate the effectiveness of risk mitigation measures, and make necessary adjustments.
# 
# ##### 1.5. Regulatory Compliance
# 
# Financial institutions must comply with applicable laws, regulations, and standards related to risk management. This includes implementing internal controls, reporting requirements, and governance frameworks to ensure compliance with regulatory guidelines.
# 
# By conducting thorough risk assessments, financial institutions can make informed decisions, protect themselves against potential losses, maintain financial stability, and safeguard the interests of their clients and stakeholders.

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# ## Market Risk
# 
# Market risk is the potential for financial losses resulting from adverse changes in market conditions, such as fluctuations in stock prices, interest rates, foreign exchange rates, or commodity prices. Market risk assessment involves evaluating the impact of these changes on investment portfolios, trading positions, or financial products. 
# 
# 1. **Risk Measurement:** Risk measurement involves quantifying and evaluating market risks to understand their potential impact. Common methods used for market risk measurement include: 
#     - **Value at Risk (VaR)**: VaR estimates the maximum potential loss of an investment or portfolio over a specified time horizon with a given level of confidence. For example, a 95% VaR of $1 million for a portfolio implies that there is a 5% chance of losing more than $1 million over the specified period.
#     - **Expected Shortfall (ES)**: ES, also known as conditional VaR, measures the expected average loss beyond the VaR. It provides additional information about the potential losses in extreme market conditions.
#     - **Stress Testing**: Stress testing involves analyzing the impact of severe market scenarios on investments or portfolios. This can include simulating market crashes, interest rate spikes, or geopolitical events to assess potential losses.
# 
# 
# 2. **Risk Mitigation:** Risk mitigation strategies aim to reduce the impact of market risks on investments. Here are some common risk mitigation techniques:
#     - **Diversification:** Spreading investments across different asset classes, sectors, or geographic regions can help reduce concentration risk. If one investment performs poorly, others may provide a buffer. 
#     - **Hedging:** Using derivative instruments like options, futures, or swaps can hedge against adverse price movements. For example, buying a put option can protect against a decline in the value of a stock. 
#     - **Dynamic Asset Allocation:** Regularly adjusting portfolio allocations based on market conditions can help reduce risk exposure. For instance, shifting investments from equities to bonds during periods of high market volatility.
# 
# 3. **Risk Monitoring:** Risk monitoring involves ongoing surveillance and analysis of market risks. It includes:
#     - **Regular Portfolio Reviews:** Periodic assessment of investment portfolios to ensure they align with the risk tolerance and investment objectives of investors.
#     - **Monitoring Market Indicators:** Tracking relevant market indicators such as stock indices, interest rates, or exchange rates to identify potential risks or market trends.
#     - **Real-time Reporting:** Utilizing technology and risk management systems to provide real-time updates on market positions and risk exposures.
# 
# 4. **Regulatory Compliance:** Regulatory compliance ensures that financial institutions adhere to applicable laws and regulations. Key aspects of regulatory compliance in managing market risk include:
#     - **Capital Adequacy:** Compliance with regulatory capital requirements to ensure sufficient capital buffers are maintained to absorb potential losses.
#     - **Reporting and Disclosure:** Providing accurate and timely reporting on risk exposures, market positions, and risk management strategies to regulatory authorities.
#     - **Risk Governance:** Establishing robust risk governance frameworks, policies, and procedures to ensure effective risk management and compliance.
# 
# To reduce risk in investments, a combination of strategies can be employed:
# - **Diversify:** Invest in a mix of asset classes (stocks, bonds, real estate) to spread risk.
# - **Set Risk Tolerance:** Determine your risk tolerance and invest accordingly. This ensures that investments align with your risk appetite.
# - **Regular Monitoring:** Keep track of market trends and portfolio performance to make informed decisions.
# - **Stay Informed:** Stay updated on economic indicators, market news, and industry developments that can impact investments.
# - **Consult with Professionals:** Seek advice from financial advisors or professionals who specialize in risk management and investment strategies.
# 
#     It is important to note that the risk cannot be eliminated entirely, but by employing prudent risk management techniques, investors can aim to minimize potential losses and optimize their investment outcomes.
# 

# %% [markdown]
# ### Value at Risk (VaR)
# Value at Risk (VaR) is a widely used risk measurement tool that quantifies the maximum potential loss in the value of an investment or portfolio over a specific time period with a given level of confidence. VaR provides an estimate of the potential downside risk and is typically expressed as a monetary value or percentage. 
# 
# ![image.png](attachment:image.png)
# 
# There are different methods to calculate VaR, including the historical method, variance-covariance method (parametric method), and Monte Carlo simulation. I'll explain the historical method and the variance-covariance method below:
# 
# 1. Historical VaR: The historical VaR is calculated based on historical returns or changes in the value of the investment or portfolio. The steps involved are:
# 
#     - **Step 1:** Determine the historical return series: Gather historical data for the investment or portfolio returns over the chosen time period (e.g., daily, monthly) that matches the desired time horizon.
#     - **Step 2:** Order the historical returns: Arrange the historical returns in descending order from the most negative to the most positive.
#     - **Step 3:** Identify the VaR level: Determine the confidence level or probability level for the VaR calculation (e.g., 95%, 99%). This represents the likelihood of the VaR being breached.
#     - **Step 4:** Calculate the VaR: Locate the historical return that corresponds to the chosen VaR level based on the ordered return series. The corresponding return represents the VaR value. To convert it to a monetary value, multiply it by the investment value.
# 
#     Example: Let's assume you have daily returns for investment over the past 1,000 trading days. You want to calculate the 95% VaR for a $100,000 investment.
#     - Step 1: Gather the historical returns data.
#     - Step 2: Order the historical returns in descending order.
#     - Step 3: Choose the VaR level (e.g., 95%).
#     - Step 4: Identify the VaR value by locating the return corresponding to the 95th percentile in the ordered return series.

# %% [markdown]
# 2. **Variance-Covariance VaR:** The variance-covariance VaR, also known as the parametric VaR, relies on assumptions of normal distribution and measures the risk based on the mean and standard deviation of returns. The steps involved are:
# - **Step 1:** Calculate the mean return and standard deviation: Calculate the average return and standard deviation of the investment or portfolio returns over the chosen time period.
# - **Step 2:** Determine the Z-score: Based on the chosen confidence level (e.g., 95%, 99%), determine the corresponding Z-score from the standard normal distribution table.
# - **Step 3:** Calculate the VaR: Multiply the Z-score by the standard deviation and subtract the result from the mean return. To convert it to a monetary value, multiply it by the investment value.
#   
# 3. **Monte Carlo Simulation:** A Monte Carlo simulation is used to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. It is a technique used to understand the impact of risk and uncertainty.
# 
#     **KEY TAKEAWAYS:**
# 
#     - A Monte Carlo simulation is a model used to predict the probability of a variety of outcomes when the potential for random variables is present.
#     - Monte Carlo simulations help to explain the impact of risk and uncertainty in prediction and forecasting models.
#     - A Monte Carlo simulation requires assigning multiple values to an uncertain variable to achieve multiple results and then averaging the results to obtain an estimate.
#     - Monte Carlo simulations assume perfectly efficient markets.
# 

# %% [markdown]
# ## What Does a Risk Analyst Do?
# 
# On a day-to-day basis, risk analysts spend their time:
# 
# - Analyzing models and data within the scope of a business action.
# - Predicting and determining the likely outcome of a business decision.
# - Preparing reports on findings and making recommendations.
# - Using analytics software and tools to calculate huge sums of data.
# - Consulting with stakeholders and informing leadership of findings.
# - Evaluating records and plans to determine the cost of actions.
# - Anticipating potential losses and proposing appropriate safeguards.

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# ## REFERENCE
# 
# - https://www.investopedia.com/terms/v/var.asp
# - https://www.investopedia.com/financial-term-dictionary-4769738
# - https://www.simtrade.fr/blog_simtrade/variance-covariance-method-var-calculation/ (must see)
# - https://www.investopedia.com/terms/m/montecarlosimulation.asp  (Monte-Carlo simulation)


