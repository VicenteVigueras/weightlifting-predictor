import numpy as np

"""
This will be used to derive the model and understand how the beta coefficients are calculated.
"""

# Model a sample equation in the form of y = β0​+β1​x (y = mx + b)
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])  

y = np.array([3, 5, 7])

# The closer beta gets to [1, 2], the better the model fits the data
beta_hardcoded = np.array([1.0, 2.0]) # <-- perfect fit
pred = X @ beta_hardcoded
print(pred)


# We model the squared error loss function to measure how wrong a prediciton is
# This way we look for when loss is minimized 
def loss(beta):
    pred = X @ beta
    return np.sum((pred - y)**2)

# Gradient Descent
beta = np.zeros(X.shape[1])

learning_rate = 0.01
for i in range(2500):
    pred = X @ beta
    grad = 2 * X.T @ (pred - y)
    beta = beta - learning_rate * grad

print("beta expected: ", beta_hardcoded)
print("beta actual: ", beta)