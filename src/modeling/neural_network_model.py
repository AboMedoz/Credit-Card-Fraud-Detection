import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Macros
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'processed')
MODELS_PATH = os.path.join(ROOT, 'models')

df = pd.read_csv(os.path.join(DATA_PATH, 'processed.csv'))


x = df.drop(columns=['Class'])
y = df['Class']

# handle class imbalance
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights / np.max(weights) * 2))

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# Scale 'Amount'
scaler = StandardScaler()
x_train['Amount'] = scaler.fit_transform(x_train[['Amount']])
x_test['Amount'] = scaler.transform(x_test[['Amount']])

model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), class_weight=class_weights)

_, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

model.save(os.path.join(MODELS_PATH, 'nn_model.keras'))