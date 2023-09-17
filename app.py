from flask import Flask, request, jsonify
import torchaudio
import io
import firebase
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
app = Flask(__name__)

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=8) 

config = {
  "apiKey": "AIzaSyBDqvrZBCVPknOSERwKDasBfekaALGVFlY",
  "authDomain": "musicgen-adf71.firebaseapp.com",
  "projectId": "musicgen-adf71",
  "storageBucket": "musicgen-adf71.appspot.com",
  "messagingSenderId": "203606423178",
  "appId": "1:203606423178:web:443d8e60da01469e0cdf73",
  "measurementId": "G-S8JQVXLQ5X",
  "databaseURL": "xxxxxx"
}

firebase = firebase.initialize_app(config)

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        data = request.get_json()
        descriptions = data.get('descriptions', [])

        if not isinstance(descriptions, list):
            descriptions = [descriptions]

        if not descriptions:
            return jsonify({'error': 'No descriptions provided'}), 400

        wav = model.generate(descriptions)

        audio_urls = []
        for idx, one_wav in enumerate(wav):
            audio_filename = f'{idx}.wav'

            audio_bytesio = io.BytesIO()
            torchaudio.save(audio_bytesio, one_wav.cpu(), model.sample_rate, format="wav")

            cloudpath = f"audios/{audio_filename}"
            firebase.storage().child(cloudpath).put(audio_bytesio.getvalue())

            audio_url = firebase.storage().child(cloudpath).get_url(None)
            audio_urls.append(audio_url)

        return jsonify({'audio_urls': audio_urls}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    