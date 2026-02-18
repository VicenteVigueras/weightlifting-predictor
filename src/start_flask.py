from flask import Flask, request, jsonify
from flask_cors import CORS
from core import predict_progress

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()
    print(f"Received: {data} ")
    age = data.get('age')
    weight = data.get('weight')
    height = data.get('height')
    experience = data.get('experience')
    goal = data.get('goal')
    exercise = data.get('exercise')
    lifted = data.get('lifted')
    reps = data.get('reps')
    sets = data.get('sets')
    progress = predict_progress(age, weight, height, experience, goal, exercise, lifted, reps, sets)
    return jsonify({'progress': f'{progress}'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)