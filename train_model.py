import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy training data: X = marketing spend, y = sales
X = np.array([[100], [200], [300], [400], [500]])
y = np.array([10, 20, 30, 40, 50])
model = LinearRegression()
model.fit(X, y)
# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
