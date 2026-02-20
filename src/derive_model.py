import numpy as np

"""
This will be used to derive the model and understand how the beta coefficients are calculated.
"""

# model a sample equation in the form of y=β0​+β1​x
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])  

y = np.array([3, 5, 7])

# conclusion - the closer beta gets to [1, 2], the better the model fits the data
beta = np.array([1.0, 2.0])
pred = X @ beta
print(pred)