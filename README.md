# 🚗 Waze User Churn & Retention Analysis

Welcome to a project where user behavior meets machine learning.  
Instead of only asking whether users stayed or left, this project asks:

## ❓ What if we could identify **which Waze users are most likely to churn** and recommend actions to retain them?

Using **Python, machine learning, and business threshold analysis**, this project explores user activity patterns, predicts churn risk, and groups users into actionable retention segments.


## 📌 What This Project Does

This project analyzes Waze user behavior data to understand why users may leave the app and how they can be retained.

The project focuses on:

- 📉 Churn rate analysis
- 📱 Device-based churn comparison
- 🚘 Driving and session behavior
- 📊 Activity-level churn patterns
- 🤖 Churn prediction using machine learning
- 🎯 Business threshold optimization
- 🧩 Risk segmentation
- 💡 Retention recommendations

Rather than stopping at model accuracy, this project connects predictions to a practical business question:

> Which users should Waze target with retention campaigns?


## 📂 Dataset

The dataset used in this project is a public Waze user churn dataset.

### It includes user-level behavioral features such as:

- User retention/churn label
- Number of sessions
- Number of drives
- Total sessions
- Days after onboarding
- Favorite navigation usage
- Kilometers driven
- Driving duration
- Activity days
- Driving days
- Device type

The target variable is whether a user was:

- **Retained**
- **Churned**

> Note: This is a public project dataset and should not be interpreted as internal Waze company data.


## 🛠️ Tech Stack

- **Python** 🐍
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Logistic Regression**
- **Random Forest**
- **Feature engineering**
- **Classification metrics**
- **Business threshold analysis**


## 🎯 Project Goals

The goal of this project was to analyze Waze user behavior and identify patterns that explain customer churn.

The main goals were to:

- Clean and prepare the churn dataset
- Explore churn patterns across user behavior
- Understand how activity level relates to churn
- Build machine learning models to predict churn risk
- Compare Logistic Regression and Random Forest models
- Identify the most important churn drivers
- Choose a business-friendly churn probability threshold
- Segment users into High, Medium, and Low Risk groups
- Recommend retention actions for each risk segment


## 🧠 Methods Used

This project combines exploratory analysis, machine learning, and business-focused decision-making.

### 1. 🧹 Data Cleaning

The dataset contained missing values in the churn label column.  
Rows without a known churn/retention label were removed because they could not be used for supervised learning.

The target was converted into a binary variable:

- `1` = churned
- `0` = retained


### 2. 📊 Exploratory Data Analysis

The project first explored the overall churn rate and compared churn patterns across different user groups.

Key EDA questions included:

- What percentage of users churned?
- Does churn differ by device type?
- Are low-activity users more likely to churn?
- How do retained and churned users differ in driving behavior?


### 3. 🧩 Activity-Based Churn Segmentation

Users were divided into activity groups based on `activity_days`.

The analysis found a strong relationship between activity level and churn:

- **Very Low Activity users:** 34.72% churn rate
- **Low Activity users:** 18.48% churn rate
- **Medium Activity users:** 9.94% churn rate
- **High Activity users:** 5.10% churn rate

This suggests that lower engagement is one of the clearest signs of churn risk.


### 4. 🛠️ Feature Engineering

Several new features were created to better describe user behavior:

- `km_per_drive`
- `minutes_per_drive`
- `drives_per_session`
- `activity_rate`
- `driving_rate`
- `total_fav_navigations`

These features helped capture user engagement, driving intensity, and navigation habits.


### 5. 🤖 Machine Learning Models

Two classification models were trained and compared:

### Logistic Regression

Used as an interpretable baseline model.

### Random Forest

Used to capture non-linear relationships between user behavior and churn.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC


## 📊 Model Results

| Model | ROC-AUC | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.744 | 0.671 | 0.312 | 0.708 | 0.433 |
| Random Forest | 0.741 | 0.717 | 0.335 | 0.606 | 0.431 |

The Logistic Regression model had a slightly higher ROC-AUC and churn recall, while the Random Forest model had better overall accuracy and churn precision.

For the business threshold and risk segmentation part of the project, the Random Forest probabilities were used.


## 🎯 Business Threshold Analysis

Instead of using the default `0.50` classification threshold, this project tested multiple churn probability thresholds from `0.20` to `0.80`.

The goal was to choose a threshold based on estimated business value.

### Business assumptions:

- Value of saving one churn-risk user: **$25**
- Cost of targeting one user with a retention campaign: **$5**

The best threshold was:

## ✅ Selected Threshold: `0.45`

At this threshold:

- **True churners caught:** 343
- **Missed churners:** 164
- **False alarms:** 724
- **Targeted users:** 1,067
- **Churn precision:** 32.1%
- **Churn recall:** 67.7%
- **Estimated net value:** $3,240

This threshold balances catching churn-risk users while avoiding excessive campaign costs.


## 🔍 Key Churn Drivers

The Random Forest model identified the most important churn prediction features as:

- Activity days
- Driving days
- Days after onboarding
- Driving rate
- Activity rate
- Total favorite navigations
- Duration minutes driven
- Favorite navigation usage
- Total sessions
- Driven kilometers

The biggest takeaway:

> Users with fewer activity days and fewer driving days are much more likely to churn.


## 🧩 Risk Segmentation

Users were grouped into three churn-risk segments based on predicted churn probability:

| Risk Segment | Users | Actual Churn Rate | Avg Churn Probability |
|---|---:|---:|---:|
| High Risk | 276 | 47.10% | 74.62% |
| Medium Risk | 985 | 25.08% | 54.86% |
| Low Risk | 1,599 | 8.13% | 23.76% |

This segmentation makes the model easier to use from a business perspective.


## 💡 Retention Recommendations

### 🔴 High Risk Users

**Recommended action:**  
Send immediate personalized retention offers or app re-engagement campaigns.

**Reason:**  
This group has the highest actual churn rate, so retention spending is most justified.


### 🟡 Medium Risk Users

**Recommended action:**  
Send educational nudges, route-quality reminders, and feature prompts.

**Reason:**  
This group has moderate churn risk, so lower-cost interventions are more appropriate.


### 🟢 Low Risk Users

**Recommended action:**  
Avoid expensive discounts. Use light engagement messaging only.

**Reason:**  
This group has low churn risk, so aggressive retention campaigns may waste budget.



## 📌 Insights from the Output

This project showed that user activity is one of the strongest signals of churn.

The overall churn rate in the dataset was **17.74%**, meaning around 1 in 6 users left the app. When users were grouped by activity level, the pattern became much clearer:

- **Very Low Activity users** had a churn rate of **34.72%**
- **Low Activity users** had a churn rate of **18.48%**
- **Medium Activity users** had a churn rate of **9.94%**
- **High Activity users** had a churn rate of only **5.10%**

This shows that users who interact with the app less often are much more likely to churn.

Device type did not show a major difference. iPhone and Android users had very similar churn rates, so device type was not a strong churn driver in this analysis.

The most important churn prediction features were:

- Activity days
- Driving days
- Days after onboarding
- Driving rate
- Activity rate
- Favorite navigation usage
- Total sessions
- Driven kilometers

The main business insight is simple:

> Users who are less active and drive less frequently are more likely to churn, so retention efforts should focus on early signs of reduced engagement.


## 🚦 Why This Project Matters

Customer churn is an important problem for any app-based business.

For a navigation app like Waze, retaining active users matters because user engagement supports the overall product experience. If users stop opening the app, stop driving with it, or stop using navigation features, they may eventually leave completely.

This project matters because it does more than just build a machine learning model. It connects the model output to a real business decision:

- Which users are most likely to churn?
- Which users should be targeted first?
- How can retention campaigns be prioritized?
- How can a company avoid spending money on users who are unlikely to leave?

By combining churn prediction with risk segmentation and business threshold analysis, this project turns model results into practical retention recommendations.


## 📊 Results Achieved

This project successfully built an end-to-end churn analysis workflow.

### Key results:

- The overall churn rate was **17.74%**
- Very Low Activity users had the highest churn rate at **34.72%**
- High Activity users had the lowest churn rate at **5.10%**
- Logistic Regression achieved a ROC-AUC of **0.744**
- Random Forest achieved a ROC-AUC of **0.741**
- Random Forest achieved better overall accuracy at **71.7%**
- The best business threshold was selected as **0.45**
- At this threshold, the model caught **343 churned users**
- The estimated net value from the retention campaign was **$3,240**

The final risk segments showed a clear difference in churn behavior:

- **High Risk users:** 47.10% actual churn rate
- **Medium Risk users:** 25.08% actual churn rate
- **Low Risk users:** 8.13% actual churn rate

This makes the model useful from a business point of view because users can be prioritized based on churn risk.


## 🚀 Future Improvements

This project can be improved in several ways:

- Add a Streamlit app to let users explore churn predictions interactively
- Build a Power BI or Tableau dashboard for business reporting
- Add SHAP analysis to explain individual user predictions
- Test more advanced models like XGBoost or LightGBM
- Tune hyperparameters using cross-validation
- Add customer lifetime value assumptions for better business impact analysis
- Compare retention strategies for different user groups
- Track churn risk over time if time-based user data becomes available
- Add more behavioral features, such as app open frequency or route search patterns


## 🎯 Note from the Author

I built this project to practice solving a realistic churn problem from both a data science and business perspective.

What I liked about this project is that it was not only about getting the best model score. The more important part was understanding what the model results mean and how they could support real decisions.

This project helped me connect data cleaning, exploratory analysis, machine learning, and business recommendations in one workflow.

In simple terms, the goal was not just to predict who might leave.

The goal was to understand why users may leave and what can be done to retain them.
