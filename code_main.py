from flask import Flask, request, jsonify
import os
import os.path as ops
import cv2
import numpy as np
from local_utils.config_utils import parse_config_utils
from models import build_sam_clip_text_ins_segmentor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def segment_image(input_image_path, insseg_cfg_path, text=None, cls_score_thresh=None, use_text_prefix=False):
    if not ops.exists(input_image_path):
        return {"error": "Input image path not exists"}

    if not ops.exists(insseg_cfg_path):
        return {"error": "Instance segmentation config path not exists"}

    insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)

    if text is not None:
        unique_labels = text.split(',')
    else:
        unique_labels = None

    if cls_score_thresh is not None:
        insseg_cfg.INS_SEG.CLS_SCORE_THRESH = cls_score_thresh

    segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)

    ret = segmentor.seg_image(input_image_path, unique_label=unique_labels, use_text_prefix=use_text_prefix)

    return ret

@app.route('/segment', methods=['POST'])
def segment_image_endpoint():
    if 'input_img' not in request.files:
        return jsonify({"error": "No input image provided"}), 400

    input_image = request.files['input_img']
    pattern_image = request.files['pattern_img']

    if not input_image or not pattern_image:
        return jsonify({"error": "Both input image and pattern image are required"}), 400

    input_image.save('input_image.jpg')
    pattern_image.save('pattern_image.jpg')

    args = {
        'input_img': 'input_image.jpg',
        'pattern_img': 'pattern_image.jpg',
        'cfg_path': './config/insseg.yaml',
        'text': request.form.get('text'),
        'cls_score_thresh': float(request.form.get('cls_score_thresh')) if request.form.get('cls_score_thresh') else None,
        'use_text_prefix': request.form.get('use_text_prefix', type=bool)
    }

    result = segment_image(**args)

    if 'error' in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
