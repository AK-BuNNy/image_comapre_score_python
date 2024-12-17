from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
from compare_sketches import compare_sketches
import cv2
import numpy as np
import os
import shutil

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create upload directory if it doesn't exist

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    try:
        # Extract files from the request
        ref = request.files.get('image1')
        cand = request.files.get('image2')

        if not ref or not cand:
            return jsonify({"error": "Both image1 and image2 are required."}), 400

        # Save uploaded files temporarily
        ref_path = os.path.join(UPLOAD_DIR, ref.filename)
        cand_path = os.path.join(UPLOAD_DIR, cand.filename)
        
        with open(ref_path, "wb") as ref_file:
            shutil.copyfileobj(ref, ref_file)
        with open(cand_path, "wb") as cand_file:
            shutil.copyfileobj(cand, cand_file)

        # Compute the similarity score
        score = compare_sketches(ref_path, cand_path)

        # Cleanup temporary files
        os.remove(ref_path)
        os.remove(cand_path)

        return jsonify({"score": score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

