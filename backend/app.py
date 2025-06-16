from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import logging
import gc
import traceback


app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # 👈 already handled by flask-cors but helps
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load BLIP model
try:
    logger.info("🧠 Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"✅ Model loaded on device: {device}")
except Exception as e:
    logger.error(f"❌ Failed to load BLIP model: {str(e)}")
    raise

IMAGE_PATH = "current.jpg"

@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        logger.info("🔁 /caption route hit")

        image_file = request.files.get('image')
        if not image_file:
            logger.error("❌ No image received")
            return jsonify({"error": "No image received"}), 400

        image = Image.open(image_file).convert('RGB')
        image.save(IMAGE_PATH)
        logger.info("📸 Image saved")

        logger.info("🧠 Generating caption...")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=128, num_beams=3)

        caption = processor.decode(output[0], skip_special_tokens=True)
        logger.info(f"✅ Caption: {caption}")

        # Cleanup
        del inputs, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({"caption": caption})

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Flask server on port 5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
