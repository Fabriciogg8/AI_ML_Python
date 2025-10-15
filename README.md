# 🧠 Kubicle Machine Learning Projects – Fabricio González Guasque

This repository contains two machine learning projects developed as part of the **Kubicle AI & Machine Learning Certification**.

Each notebook applies predictive modeling techniques using **Python, Pandas, and Scikit-Learn**, solving real-world business problems through data-driven insights.

---

## 🏦 Project 1: Predicting Loan Default — Securibank Dataset

📘 Certification Module: *Identify Risk of Default with Predictive Analytics*
🗂️ File: `loan_default.ipynb`

### 🔍 Objective

To help **Securibank** identify customers most likely to default on their loan payments using machine learning models trained on historical financial data.

### ✅ Key Steps

* **Data Understanding:**
  Load and explore customer demographics, repayment history, and outstanding balances.
* **Feature Engineering:**
  Encoded categorical variables and split data into training and test sets (85% / 15%).
* **Modeling:**
  Built and compared multiple classification algorithms:

  * 🌳 Decision Tree Classifier
  * 🧮 Gaussian Naive Bayes
  * ⚙️ Support Vector Machine (SVM)
* **Evaluation:**
  Used confusion matrices, precision, recall, and F1-scores to assess performance.
* **Insights:**

  * Decision Tree achieved ~82% accuracy but underperformed for true defaulters.
  * Naive Bayes detected more defaulters but had low overall precision.
  * SVM struggled with class imbalance, misclassifying most defaulters.

### 📊 Key Findings

| Model                       | Accuracy | Strengths                      | Weaknesses                 |
| --------------------------- | -------- | ------------------------------ | -------------------------- |
| Decision Tree (max_depth=1) | ~82%     | Simple, interpretable          | Underfits complex data     |
| Gaussian Naive Bayes        | ~42%     | Fast, handles categorical data | Low accuracy               |
| Support Vector Machine      | ~78%     | Strong theoretical performance | Fails with imbalanced data |

### 💡 Business Recommendations

* Implement **Decision Trees** as the baseline model for risk detection.
* Improve detection of defaulters using **class weighting** and **resampling techniques**.
* Integrate model insights into Securibank’s **credit scoring process** to reduce financial risk.

---

## 🚗 Project 2: Improving Car Pricing Strategy — Akashi Motors

📘 Certification Module: *Machine Learning — Regression Models*
🗂️ File: `car_prices.ipynb`

### 🔍 Objective

Assist **Akashi Motors**, a Japanese automaker, in redesigning its U.S. pricing strategy by predicting optimal car prices through regression models.

### ✅ Key Steps

* **Data Preparation:**
  Loaded and explored U.S. car sales data, selecting relevant numeric and categorical features.
* **Feature Engineering:**
  Removed redundant variables and applied one-hot encoding for categorical features.
* **Modeling:**

  * Built a **Linear Regression** model and evaluated its assumptions.
  * Identified heteroscedasticity in residuals → tested **non-parametric models**:

    * 🌳 Decision Tree Regressor
    * 🧭 K-Nearest Neighbors (KNN)
    * ⚙️ Support Vector Regression (SVR)
* **Evaluation:**
  Compared model performance using **Mean Absolute Error (MAE)**.

### 📊 Model Performance

| Model         | MAE (USD) | Strengths                       | Weaknesses                   |
| ------------- | --------- | ------------------------------- | ---------------------------- |
| Decision Tree | 2,082     | Best performance, interpretable | May overfit                  |
| KNN           | 2,258     | Simple, robust                  | Sensitive to feature scaling |
| SVR           | 5,706     | Handles non-linearity           | High computational cost      |

### 💡 Business Recommendations

* Adopt the **Decision Tree Regressor** as the baseline pricing model.
* Collect additional market data (brand perception, safety ratings) to enhance predictions.
* Explore **ensemble methods** (Random Forest, Gradient Boosting) for improved accuracy.
* Deploy a **pricing dashboard** for strategic decision-making.

---

## 🧰 Tech Stack

* Python (Pandas, NumPy, Matplotlib, Seaborn)
* Scikit-Learn (Linear Regression, Decision Trees, Naive Bayes, SVM, KNN, SVR)
* Google Colab for notebook execution
* Data visualization and model evaluation metrics

---

## 🧑‍💻 About Me

**Fabricio González Guasque**  
Python Developer & Data Analyst with a background in Agronomic Engineering.  
🔗 [LinkedIn](https://www.linkedin.com/in/fabriciogonzalezguasque) | [GitHub](https://github.com/Fabriciogg8)

---

## 📜 Certification

These projects were developed as part of the **AI & ML with Python** certification

Courses completed:
- AI Fundamentals Module
  -  Introduction to Artificial Intelligence
  -  Business Applications of AI
  -  Predicting Future Values
  -  Predicting Scenarios
  -  Identifying Patterns
- Machine Learning with Python Module
  - Regression Analysis in Python
  - Improve a car company’s pricing strategy
  - Decision Trees in Python
  - K-Means Clustering in Python
  - Advanced Regression
  - Advanced Classification
  - Identify risk of default with predictive analytics
  - Advanced Clustering
