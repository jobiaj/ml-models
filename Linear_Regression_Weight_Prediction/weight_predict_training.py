import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example dataset
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 65, 75, 85]
}

df = pd.DataFrame(data)
X = df[['Height']]  # input feature
y = df['Weight']    # target output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predicted weights:", predictions)
print("Predicted weight for 175 cm:", model.predict([[175]]))

