import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression

data_frame = pd.read_csv('creditcard.csv')

v = []
for i in range(1, 29):
    v.append(f'V{i}')

x = data_frame[[*v]]
y = data_frame['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict) * 100
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_predict)
print(report)

y_predict_proba = model.predict_proba(x_test)
fpr, tpr, threshold = roc_curve(y_test, y_predict_proba)
roc = auc(fpr, tpr)

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
sns.set_style('darkgrid')
sns.lineplot(x=fpr, y=tpr, label=f'AUC: {roc:.2f}', lw=2)
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.legend(loc='lower right')
plt.show()