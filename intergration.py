from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = "uploaded_audio.wav"
    file.save(file_path)
    
    gender = predict_gender(model, file_path)
    if gender != 'female':
        return jsonify({"error": "Please upload a female voice."})
    
    features = extract_features(file_path)
    prediction = model.predict(features)
    emotion = np.argmax(prediction)
    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)
