import torchaudio
import torch
import torchaudio.transforms as T

def load_model():
    model = torch.hub.load('pyannote/pyannote-audio', 'emb')
    return model

def predict_gender(model, file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    embedding = model(waveform)
    gender = classify_gender(embedding)
    return gender

model = load_model()
gender = predict_gender(model, "path_to_audio_file.wav")
if gender != 'female':
    print("Please upload a female voice.")
