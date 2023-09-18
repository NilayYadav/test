from flask import Flask, request, jsonify
from transformers import AutoProcessor, MusicgenForConditionalGeneration

app = Flask(__name__)

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

@app.route('/generate_music', methods=['POST'])
def generate_music():
    try:
        # Get the input text from the request body
        data = request.json
        input_text = data.get('input_text')

        # Check if input_text is provided
        if not input_text:
            return jsonify({"error": "Input text is missing"}), 400

        # Tokenize the input text
        inputs = processor(
            text=input_text,
            padding=True,
            return_tensors="pt",
        )

        # Generate music
        audio_values = model.generate(**inputs, max_new_tokens=256)
        
        # Return the generated audio values
        return jsonify({"audio_values"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
