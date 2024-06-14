import base64
import os
import os.path as ops
import cv2
from PIL import Image
import numpy as np
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_sam_clip_text_ins_segmentor
from download import download_file
LOG = init_logger.get_logger('instance_seg.log')
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def instance_segmentation(input_img, pattern_image_name):
    # Convert input image to numpy array
    input_img = np.array(input_img)
    
    # Init args
    input_image_path = 'input_image.png'
    cv2.imwrite(input_image_path, input_img)

    if not ops.exists(input_image_path):
        return "Error: Input image path does not exist."

    # Init segmentor
    insseg_cfg_path = './config/insseg.yaml'
    if not ops.exists(insseg_cfg_path):
        return "Error: Instance segmentation config path does not exist."
    insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)
    unique_labels = None
    use_text_prefix = False
    segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)

    # Segment input image
    ret = segmentor.seg_image(input_image_path, unique_label=unique_labels, use_text_prefix=use_text_prefix)

    # Save results
    save_dir = './output/insseg'
    os.makedirs(save_dir, exist_ok=True)
    ori_image_save_path = ops.join(save_dir, 'input_image.png')
    cv2.imwrite(ori_image_save_path, ret['source'])
    Largest_mask_add_save_path = ops.join(save_dir, 'largest_segment_mask.png')
    wall_mask_image = cv2.imread(ret['largest_segment_mask_path'])
    cv2.imwrite(Largest_mask_add_save_path, wall_mask_image)

    # Load images
    original_image = cv2.imread(ori_image_save_path)
    mask_image = cv2.imread(Largest_mask_add_save_path)

    # Convert mask image to grayscale
    mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask_gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask_gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Download the specified pattern image from S3
    pattern_images_folder = 's3_bucket_pattern_images'
    pattern_image_path = os.path.join(pattern_images_folder, pattern_image_name)
    download_file('decorazzi-visualizer', pattern_image_name, pattern_image_path)

    # Process input image with the pattern image
    pattern_image = cv2.imread(pattern_image_path)
    pattern_image_resized = cv2.resize(pattern_image, (original_image.shape[1], original_image.shape[0]))

    # Overlay pattern image onto original image using mask
    result = original_image.copy()
    result[mask == 255] = pattern_image_resized[mask == 255]

    # Cleanup
    os.remove(input_image_path)
    os.remove(ori_image_save_path)
    os.remove(Largest_mask_add_save_path)

    return result

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files or 'pattern_image_name' not in request.form:
        return jsonify({'error': 'No image or pattern image name sent'}), 400

    file = request.files['image']
    pattern_image_name = request.form['pattern_image_name']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save the image temporarily
        image_path = 'temp_input_image.png'
        file.save(image_path)

        # Perform instance segmentation
        try:
            result_image = instance_segmentation(cv2.imread(image_path), pattern_image_name)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        # Encode the image data as base64
        _, buffer = cv2.imencode('.png', result_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Delete the temporary image file
        os.remove(image_path)

        return jsonify({'result_image': encoded_image}), 200

    return jsonify({'error': 'Unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
