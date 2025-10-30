import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
data_frame = pd.read_csv('dataset.csv')
print(data_frame)

# Features innit
v = []
for i in range(1, 29):
    v.append(f'V{i}')

# Dataset needn't cleaning can select Features rn
x = data_frame[[*v]]
y = data_frame['Class']

# Checking how imbalanced the Dataset is
sns.countplot(x=y)
plt.title('Fraud vs Non-Fraud')
plt.xlabel('0 = Non-Fraud, 1 = Fraud')
plt.ylabel('Count')
plt.show()
# Dataset is heavily unbalanced might consider using SMOTE

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# Scaling for Logistic Regression
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# using Logistic Regression (Fraud = 1, Non-Fraud = 0)
model = LogisticRegression(class_weight='balanced')
model.fit(x_train_scaled, y_train)

y_predict = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)[:, -1]

accuracy = accuracy_score(y_test, y_predict) * 100
print(f"Accuracy: {accuracy:.2f}")

classification_rep = classification_report(y_test, y_predict)
print(classification_rep)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc = auc(fpr, tpr)

sns.set_style('darkgrid')

plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
sns.lineplot(x=fpr, y=tpr, label=f'AUC: {roc:.2f}', lw=2)
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.legend(loc='lower right')
plt.show()
