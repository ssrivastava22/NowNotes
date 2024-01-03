from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from PIL import Image
from detection_processing import detection, preprocess_image
import argparse
import webbrowser
from tensorflow.keras.models import load_model
from textblob import TextBlob
from doc import delete_toProcess_images, delete_uploaded_images, delete_file_content, create_google_doc, read_file_content
from preprocessing import add_padding, too_thick_check, too_thin_check, process_image, rescale, get_background_color

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESS_FOLDER'] = 'toProcess'

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Preprocess image for segmentation and perform segmentation on words
        background_color = preprocessing_for_segmentation(image_path, image_path)
        segmentation(image_path, background_color)
        
        dir_path = 'C:\\Users\\veena\\Desktop\\GeorgiaTech\\NowNotes\\src\\toProcess'
        image_names = os.listdir(dir_path)

        timestamps = [(os.path.getmtime(os.path.join(dir_path, image_name)), image_name) for image_name in image_names]
        sorted_timestamps = sorted(timestamps)

        for timestamp, image_name in sorted_timestamps:
            image_path = os.path.join(dir_path, image_name)
            print(image_path)
            
            ap = argparse.ArgumentParser()
            ap.add_argument("-i", "--image", required=True, help="path to image")
            ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained handwriting recognition model")
            args_list = ["--image", image_path, "--model", "/Users/veena/Desktop/GeorgiaTech/NowNotes/src/digitspt2.h5"]
            args = vars(ap.parse_args(args_list))

            # Load the handwriting OCR model
            print("[INFO] loading handwriting OCR model...")
            model = load_model(args["model"])
            image = cv2.imread(args["image"])
            boxes = detection(image)

            # Load and preprocess the image
            image = cv2.imread(args["image"])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect bounding boxes
            boxes = detection(image)

            # Map numbers to letter
            alphabets = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
                        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
                        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
                        36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
                        46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
                        56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}
            
            # Perform OCR
            results = []
            for box in boxes:
                x, y, x_w, y_h = box
                roi = gray[y:y_h, x:x_w]
                preprocessed_roi = preprocess_image(roi)
                predictions = model.predict(np.array([preprocessed_roi]))
                predicted_label = np.argmax(predictions)
                letter = alphabets[predicted_label]
                results.append((box, letter))

            result_string = ""
            # Process the OCR results
            textFile = 'paragraph.txt'
            # Sort detected letters by x-coordinate
            sorted_results = sorted(results, key=lambda item: item[0][0])
            for box, letter in sorted_results:
                x, y, x_w, y_h = box
                print(f"Predicted Letter: {letter}, Bounding Box: ({x}, {y}), ({x_w}, {y_h})")
                result_string += letter
    
            print(result_string)
            write_to_file(textFile, result_string + " ")
        
        filename = 'paragraph.txt'
        text = read_file_content('paragraph.txt')
        print(text)

        # # Auto-correct on final string
        # b = TextBlob(text)
        # text = b.correct().string
        # print("autocorrected = ", text)

        # Open Google Doc
        doc_url = create_google_doc(text)
        webbrowser.open(doc_url)

        # Delete existing text in paragraph.txt, images in uploads folder, and images in  folder
        delete_file_content(filename)
        delete_uploaded_images()
        delete_toProcess_images()
        return jsonify({'success': True, 'filename': filename}), 200
    else:
        return jsonify({'error': 'Failed to upload file'}), 500

def preprocessing_for_segmentation(image_path, new_path):
  background_color = get_background_color(image_path)
  image = cv2.imread(image_path)
  padded_img = add_padding(image)
  correct_thickness_img = process_image(padded_img)
  pil_img = Image.fromarray(correct_thickness_img)
  rescaled_img = rescale(pil_img)
  rescaled_img.save(new_path)
  return background_color

def segmentation(image_path, background_color):
  coords = get_text_coords(image_path)
  extract_words(image_path, coords, background_color)
  return coords

def get_text_coords(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    coords = [(result[1], result[0]) for result in results]
    return coords

def extract_words(image_path, word_boxes, background_color=(0, 0, 0), padding=50):
    image = Image.open(image_path)
    for word, box in word_boxes:
        left, top = box[0]
        right, bottom = box[2]
        left -= padding
        top -= padding
        right += padding
        bottom += padding
        left = max(0, left)
        top = max(0, top)
        right = min(right, image.width)
        bottom = min(bottom, image.height)
        word_image = image.crop((left, top, right, bottom))
        new_image = Image.new("RGB", (word_image.width + 2 * padding, word_image.height + 2 * padding), background_color)
        offset = ((new_image.width - word_image.width) // 2, (new_image.height - word_image.height) // 2)
        new_image.paste(word_image, offset)
        new_image.save(f"C:\\Users\\veena\\Desktop\\GeorgiaTech\\NowNotes\\src\\toProcess\\{word}.png")

def write_to_file(filename, text):
    try:
        with open(filename, 'r') as file:
            existing_content = file.read()
        with open(filename, 'w') as file:
            file.write(existing_content + text)
    except FileNotFoundError:
        with open(filename, 'w') as file:
            file.write(text)

if __name__ == '__main__':
    app.run(debug=True, port=3001)
