import os
import os.path as ops
import cv2
import gradio as gr
from PIL import Image
import numpy as np
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_sam_clip_text_ins_segmentor
from download import download_all_files
LOG = init_logger.get_logger('instance_seg.log')

def instance_segmentation(input_img):
    # Specify the S3 bucket name
    bucket_name = 'decorazzi-visualizer'
    # Specify the local directory to save downloaded files
    local_directory = './s3_bucket_pattern_images'

    # Create local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)
    download_all_files(bucket_name, local_directory)
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

    # Load pattern images from s3_bucket_pattern_images folder
    pattern_images_folder = 's3_bucket_pattern_images'
    pattern_images = os.listdir(pattern_images_folder)

    # Process input image with each pattern image
    result_images = []
    for pattern_image_name in pattern_images:
        pattern_image_path = os.path.join(pattern_images_folder, pattern_image_name)
        pattern_image = cv2.imread(pattern_image_path)
        pattern_image_resized = cv2.resize(pattern_image, (original_image.shape[1], original_image.shape[0]))

        # Overlay pattern image onto original image using mask
        result = original_image.copy()
        result[mask == 255] = pattern_image_resized[mask == 255]
        result_images.append(result)

    # Cleanup
    os.remove(input_image_path)
    os.remove(ori_image_save_path)
    os.remove(Largest_mask_add_save_path)

    return result_images

# Define Gradio interface
input = gr.Image(type="pil", label="Input Image")
pattern_images_folder = 's3_bucket_pattern_images'
pattern_images = os.listdir(pattern_images_folder)
outputs = [gr.Image(type="pil", label=f"Result Image {i+1}") for i in range(len(pattern_images))]

gr.Interface(instance_segmentation, input, outputs, title="AI Replace walls with wallpaper Tool", 
             description="Replace the wall segment in the input image with various pattern images.").launch(share=True)   