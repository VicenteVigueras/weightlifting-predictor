import pandas as pd
from core import predict_progress


df = pd.read_csv("../resources/sample_test.csv")

def test_model():
    predictions = []
    actual = []
    for i in range(len(df)):
        age = df.loc[i, 'Age']
        weight = df.loc[i, 'Weight']
        height = df.loc[i, 'Height']
        experience = df.loc[i, 'Experience']
        goal = df.loc[i, 'Goal']
        exercise = df.loc[i, 'Exercise']
        lifted = df.loc[i, 'WeightLifted']
        reps = df.loc[i, 'Reps']
        sets = df.loc[i, 'Sets']
        progress = predict_progress(age, weight, height, experience, goal, exercise, lifted, reps, sets)
        predictions.append(int(progress))

    for i in range(len(df)):
        progress = df.loc[i, 'Progress']
        actual.append(int(progress))

    return predictions, actual

def average(predictions, actual):
    accuracy = []
    total_error = 0
    count = 0
    for i in range(len(predictions)):
        numerator = abs(actual[i] - predictions[i])
        denominator = abs(actual[i])
        result = (numerator / denominator) * 100
        if result == 0:
            count = count + 1
        accuracy.append(result)
        total_error += result

    average_error = total_error / len(predictions)
    print(average_error)
    print(accuracy)
    print(count)
    return accuracy, average_error


predictions, actual = test_model()
average(predictions,actual)