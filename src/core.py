import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("../resources/sample.csv")

label_encoder_experience = LabelEncoder()
label_encoder_goal = LabelEncoder()
label_encoder_exercise = LabelEncoder()

previous_df = df[['Exercise']]
df['Experience'] = label_encoder_experience.fit_transform(df['Experience'])
df['Goal'] = label_encoder_goal.fit_transform(df['Goal'])
df['Exercise'] = label_encoder_exercise.fit_transform(df['Exercise'])

X = df.drop(columns=['Progress'])
y = df['Progress']

scaler = StandardScaler()
X[['Age', 'Weight', 'Height', 'WeightLifted', 'Reps', 'Sets']] \
    = scaler.fit_transform(X[['Age', 'Weight', 'Height', 'WeightLifted', 'Reps', 'Sets']])

X = np.c_[np.ones(X.shape[0]), X]
y = np.array(y)

beta = np.linalg.inv(X.T @ X) @ (X.T @ y)


def predict_progress(age, weight, height, experience, goal, exercise, weight_lifted, reps, sets):
    input_data = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Height': [height],
        'Experience': [experience],
        'Goal': [goal],
        'Exercise': [exercise],
        'WeightLifted': [weight_lifted],
        'Reps': [reps],
        'Sets': [sets]
    })

    input_data['Experience'] = label_encoder_experience.transform(input_data['Experience'])
    input_data['Goal'] = label_encoder_goal.transform(input_data['Goal'])
    input_data['Exercise'] = label_encoder_exercise.transform(input_data['Exercise'])

    scaled_features = scaler.transform(
        input_data[['Age','Weight','Height','WeightLifted','Reps','Sets']]
    )
    input_data[['Age','Weight','Height','WeightLifted','Reps','Sets']] = scaled_features

    input_data = np.c_[np.ones(input_data.shape[0]),
        input_data[['Age','Weight','Height','Experience','Goal','Exercise','WeightLifted','Reps','Sets']]
    ]

    predicted_progress = input_data.dot(beta)
    return predicted_progress[0]

if __name__ == "__main__":

    print("=" * 50)
    print("\u0332".join("\n1. Label Encoding:\n"))
    print("\u0332".join("1a. Before Encoding:\n"))
    print(previous_df.head())
    print("\n")
    print("\u0332".join("1b. After Encoding:\n"))
    print(df[['Exercise']].head())
    print("=" * 50 + "\n")

    print("\u0332".join("2. Define Variables:\n"))
    print("Independent variables: \n - ", X[:,1:].shape, "\n")
    print("Dependent variable:\n - Progress\n")
    print("=" * 50)

    progress = predict_progress(
        23,75,175,"Beginner","Hypertrophy","Bench Press",45,10,4
    )

    print("\u0332".join("\n3. Predicted Progress:"))
    print(" - Expected: 5 \n - Actual: ", progress)
    print("\n", "=" * 50)
