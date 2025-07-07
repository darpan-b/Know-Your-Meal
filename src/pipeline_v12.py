import argparse
import json
import logging
from pathlib import Path
import timm
import random
import cv2
import numpy as np
import random
import subprocess
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms, models
from torchvision.ops import box_convert
import re
import supervision as sv
import pycocotools.mask as mask_util
import io
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pdb import set_trace as stx
from classification import PEClassifier
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
from config import ExperimentConfig
from scipy.spatial.distance import cdist

import json
# import google.generativeai as genai
# from google.generativeai.types import Part, Content, GenerationConfig, FinishReason # Added FinishReason for detailed logging
from sklearn.metrics.pairwise import cosine_similarity

from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import shutil
import time

from google import genai  # Use this as per the new documentation
# Import for types like Content, Part, GenerateContentConfig
from google.genai import types  # Use this as per the new documentation


import yaml
from types import SimpleNamespace


class FoodSeg:

    def __init__(self):
        """
        Initializes the pipeline by setting up logging, loading configuration,
        initializing models, and setting the current date.
        """
        # Set up logger
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(console_handler)

        # Load configuration from YAML file
        self.args = self.load_config(
            "/home/darpan/pipeline_code_darpan_local_copy/v12/config.yaml")

        # Set device (CPU or GPU)
        self.device = torch.device(self.args.device)

        # Load models and preprocessing tools onto device
        self.pe_model, self.preprocess, self.pred2, self.processor, self.grounding_model = self.load_models_on_gpu(
            self.args, self.device)

        # Set current date
        self.curdate = "20250329"

    def load_config(self, config_path):
        """
        Loads configuration from a YAML file and returns it as a SimpleNamespace object.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            SimpleNamespace: Configuration parameters loaded from the file.
        """
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return SimpleNamespace(**cfg_dict)

    def load_pe_model(self, device, config="PE-Core-L14-336"):
        """
        Loads the PE-Core vision-language model along with its preprocessing components.

        Args:
            device (torch.device): The device (CPU or GPU) to load the model onto.
            config (str): Model configuration name. Defaults to "PE-Core-L14-336".

        Returns:
            tuple: A tuple containing the model, image preprocessing transform, 
                and text tokenizer.
        """
        # Instantiate the model and move it to the specified device
        model = pe.CLIP.from_config(config, pretrained=True).to(device).eval()

        # Get image and text preprocessing functions
        preprocess = pe_transforms.get_image_transform(model.image_size)
        text_tokenizer = pe_transforms.get_text_tokenizer(model.context_length)

        return model, preprocess, text_tokenizer

    def load_models_on_gpu(self, args, device):
        """
        Loads and initializes all models used in the pipeline on the specified device.

        Args:
            args (Namespace): Configuration arguments.
            device (torch.device): The device (CPU or GPU) to load the models onto.

        Returns:
            tuple: A tuple containing:
                - PE model
                - Image preprocessing transform
                - SAM2 image predictor
                - Grounding DINO processor
                - Grounding DINO model
        """
        # Load PE-Core model and preprocessing
        pe_model, preprocess, _ = self.load_pe_model(device)

        # Build SAM2-based image predictor
        sam = build_sam2(args.sam_cfg, args.sam_ckpt, device)
        pred2 = SAM2ImagePredictor(sam)

        # Load Grounding DINO model and processor
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(device)

        return pe_model, preprocess, pred2, processor, grounding_model

    def classify_regions_pe_no_text(self,
                                    masks: np.ndarray,
                                    img_rgb: np.ndarray,
                                    preprocess,
                                    pe_model,
                                    device,
                                    proto_embs: dict[str, np.ndarray],
                                    sim_thr: float):
        """
        Classifies image regions based on visual similarity to prototype embeddings using the PE model.

        Args:
            masks (np.ndarray): Binary masks for segmented regions (H x W or N x H x W).
            img_rgb (np.ndarray): Original RGB image as a NumPy array (H x W x 3).
            preprocess (Callable): Preprocessing function for image input to the model.
            pe_model: Pretrained PE model with an encode_image() method.
            device (torch.device): Device to run inference on.
            proto_embs (dict[str, np.ndarray]): Dictionary mapping class names to prototype embeddings.
            sim_thr (float): Similarity threshold for classification.

        Returns:
            tuple:
                - class_masks (dict[str, np.ndarray]): Binary masks per class where regions are classified.
                - confs (dict[str, float]): Confidence scores per class.
        """
        H, W = img_rgb.shape[:2]
        class_masks = {c: np.zeros((H, W), bool) for c in proto_embs}
        confs = {}

        for i in range(masks.shape[0] if masks.ndim > 2 else 1):
            mask = masks[i] if masks.ndim > 2 else masks
            ys, xs = mask.nonzero()
            if ys.size == 0:
                continue

            # Crop region from original image
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            crop = img_rgb[y0:y1, x0:x1]
            crop_pil = Image.fromarray(crop)
            img_t = preprocess(crop_pil).unsqueeze(0).to(device)

            # Encode the image region using PE model
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                img_feat = pe_model.encode_image(img_t)  # shape: (1, D)

            # Normalize feature vector
            img_feat = img_feat.float()
            feat = img_feat.cpu().numpy().reshape(-1)
            feat /= (np.linalg.norm(feat) + 1e-8)

            # Compute cosine similarity with each class prototype
            sims = {
                cls: float(cosine_similarity(
                    feat[None], proto_embs[cls][None])[0, 0])
                for cls in proto_embs
            }

            # Choose the class with highest similarity if above threshold
            pred, best = max(sims.items(), key=lambda kv: kv[1])
            if best >= sim_thr:
                class_masks[pred] |= mask
                confs[pred] = max(confs.get(pred, 0.0), best)

        return class_masks, confs

    def load_prototype_embeddings(self, embeddings_dir: str, date_key: str):
        """
        Loads prototype embeddings for each class from disk.

        Args:
            embeddings_dir (str): Base directory containing embeddings organized by date.
            date_key (str): Subdirectory key in the format YYYYMMDD.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping class names to embedding vectors.

        Raises:
            FileNotFoundError: If the directory for the given date_key does not exist.
        """
        print("Date key is", date_key)
        proto_embs = {}
        dirpath = os.path.join(embeddings_dir, date_key)
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f"No embeddings for date {date_key}")

        for fn in os.listdir(dirpath):
            if fn.endswith(".npy"):
                cls = fn[:-4]  # Remove .npy extension
                proto_embs[cls] = np.load(os.path.join(dirpath, fn))

        return proto_embs

    def load_colormap(self, csv_path):
        """
        Loads a colormap from a CSV file where each row defines a class and its RGB color.

        Args:
            csv_path (str): Path to the CSV file. Must contain 'category', 'R', 'G', and 'B' columns.

        Returns:
            tuple:
                - classes (list[str]): List of class names.
                - cmap (dict[str, tuple[int, int, int]]): Mapping from class name to RGB color tuple.
        """
        import csv
        classes, cmap = [], {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cls = row["category"]
                rgb = (int(row["R"]), int(row["G"]), int(row["B"]))
                classes.append(cls)
                cmap[cls] = rgb
        self.logger.info("Loaded %d classes from colormap", len(classes))
        return classes, cmap

    def load_prototype_features(self, proto_dir, classes, enc, tf, device, max_per_class=-1):
        """
        Loads and encodes prototype images into feature vectors for each class.

        Args:
            proto_dir (str or Path): Root directory containing per-class prototype images.
            classes (list[str]): List of class names.
            enc (Callable): Feature encoder (e.g., model.encode_image).
            tf (Callable): Transform/preprocessing function for input images.
            device (torch.device): Device to perform encoding on.
            max_per_class (int): Maximum number of images to use per class (-1 for all).

        Returns:
            tuple:
                - all_feats (np.ndarray): Array of normalized feature vectors (N x D).
                - all_labels (list[str]): Corresponding class labels for each feature.
        """
        all_feats, all_labels = [], []
        with torch.no_grad():
            for cls in classes:
                pdir = Path(proto_dir) / cls
                if not pdir.is_dir():
                    self.logger.warning("No prototype folder for '%s'", cls)
                    continue

                # Gather prototype images
                imgs = sorted(pdir.glob("*.*"))
                if max_per_class > 0:
                    imgs = imgs[:max_per_class]

                for p in imgs:
                    arr = np.array(Image.open(p).convert("RGB"))
                    inp = tf(arr).unsqueeze(0).to(device)
                    feat = enc(inp).squeeze().cpu().numpy()
                    # Normalize feature vector
                    feat /= (np.linalg.norm(feat) + 1e-8)
                    all_feats.append(feat.astype("float32"))
                    all_labels.append(cls)

        self.logger.info("Embedded %d prototype crops across %d classes",
                         len(all_feats), len(set(all_labels)))
        return np.stack(all_feats, 0), all_labels

    def single_mask_to_rle(self, mask: np.ndarray) -> dict:
        """
        Converts a binary mask (2D or 3D with a single channel) into COCO RLE (Run-Length Encoding) format.

        Args:
            mask (np.ndarray): A boolean or binary mask of shape (H, W) or (H, W, 1).

        Returns:
            dict: COCO-compatible RLE representation of the mask.

        Raises:
            ValueError: If the mask is not 2D or (H, W, 1).
        """
        if mask.ndim == 2:
            mask_fortran = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_util.encode(mask_fortran)
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask_fortran = np.asfortranarray(mask.astype(np.uint8))
            rle = mask_util.encode(mask_fortran)[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle
        else:
            raise ValueError(
                f"Mask must be 2D or 3D with last dim 1, got shape {mask.shape}"
            )

    def clean_and_smart_title(self, text: str) -> str:
        """
        Converts a given string into a title-like format while preserving lowercase
        for common short words (unless at the beginning).

        Args:
            text (str): Input text string to format.

        Returns:
            str: Smartly capitalized title string.
        """
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for',
                       'nor', 'on', 'at', 'to', 'from', 'by', 'of', 'in', 'with'}

        # Replace hyphens with spaces and split into words
        words = text.replace('-', ' ').split()

        # Capitalize each word appropriately
        titled_words = [
            word.capitalize() if i == 0 or word.lower() not in small_words else word.lower()
            for i, word in enumerate(words)
        ]

        return ' '.join(titled_words)

    def zoom_masked_area(self, image: np.ndarray, mask: np.ndarray, label_text: str, output_path: Path,
                         zoom_factor=1.5, alpha_bg=0.3):
        """
        Highlights a masked region from an image by zooming it and overlaying it on a faded background.
        Adds a smartly-positioned label near the zoomed region.

        Args:
            image (np.ndarray): Original BGR image.
            mask (np.ndarray): Binary mask of the object to zoom.
            label_text (str): Label to annotate near the zoomed area.
            output_path (Path): Destination path for saving the result.
            zoom_factor (float, optional): Scaling factor for the zoom. Defaults to 1.5.
            alpha_bg (float, optional): Transparency factor for fading the background. Defaults to 0.3.

        Returns:
            PIL.Image: Final annotated image resized to original dimensions.

        Raises:
            ValueError: If image or mask is missing, or if mask contains no foreground.
        """
        if image is None or mask is None:
            raise ValueError("Image or mask is None.")

        # Convert mask to uint8 if it's boolean or normalized
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if not np.any(mask):
            raise ValueError("Mask does not contain any foreground.")

        # Fade original image background
        faded_image = (image * alpha_bg).astype(np.uint8)

        # Convert both original and faded images to RGBA
        image_rgba = Image.fromarray(cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        faded_rgba = Image.fromarray(cv2.cvtColor(
            faded_image, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # Extract masked area with transparency
        mask_pil = Image.fromarray(mask).convert("L")
        masked_area = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
        masked_area.paste(image_rgba, (0, 0), mask_pil)

        # Crop and zoom the masked area
        ys, xs = np.where(mask > 0)
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        cropped_masked = masked_area.crop((x_min, y_min, x_max + 1, y_max + 1))
        zoomed_masked = cropped_masked.resize(
            (int(cropped_masked.width * zoom_factor),
             int(cropped_masked.height * zoom_factor)),
            resample=Image.BICUBIC
        )

        # Center the zoomed region on the faded image
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        start_x = max(center_x - zoomed_masked.width // 2, 0)
        start_y = max(center_y - zoomed_masked.height // 2, 0)
        faded_rgba.paste(zoomed_masked, (start_x, start_y), zoomed_masked)

        draw = ImageDraw.Draw(faded_rgba)

        # Setup font for label
        font_size = max(24, image.shape[1] // 25)
        font = None
        narrow_font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        ]
        for font_path in narrow_font_paths:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        label_text = self.clean_and_smart_title(label_text)
        bbox = font.getbbox(label_text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Coordinates of the zoomed region
        zoom_x1 = start_x
        zoom_y1 = start_y
        zoom_x2 = start_x + zoomed_masked.width
        zoom_y2 = start_y + zoomed_masked.height

        placed = False

        # Try placing label below zoomed area
        text_y = zoom_y2 + 5
        if text_y + text_h <= faded_rgba.height + 5:
            for offset in range(0, faded_rgba.width - text_w + 1, 5):
                for text_x in [zoom_x1 + (zoomed_masked.width - text_w) // 2 - offset,
                               zoom_x1 + (zoomed_masked.width - text_w) // 2 + offset]:
                    if 0 <= text_x <= faded_rgba.width - text_w:
                        draw.text((text_x, text_y), label_text,
                                  fill="white", font=font)
                        placed = True
                        break
                if placed:
                    break

        # Try placing label above zoomed area
        if not placed:
            text_y = zoom_y1 - text_h - 5
            if text_y >= 5:
                for offset in range(0, faded_rgba.width - text_w + 1, 5):
                    for text_x in [zoom_x1 + (zoomed_masked.width - text_w) // 2 - offset,
                                   zoom_x1 + (zoomed_masked.width - text_w) // 2 + offset]:
                        if 0 <= text_x <= faded_rgba.width - text_w:
                            draw.text((text_x, text_y), label_text,
                                      fill="white", font=font)
                            placed = True
                            break
                    if placed:
                        break

        # Try placing label to the right of zoomed area
        if not placed:
            text_x = zoom_x2 + 5
            text_y = zoom_y1 + (zoomed_masked.height - text_h) // 2
            if text_x + text_w <= faded_rgba.width and 0 <= text_y <= faded_rgba.height - text_h:
                draw.text((text_x, text_y), label_text,
                          fill="white", font=font)
                placed = True

        # Try placing label to the left of zoomed area
        if not placed:
            text_x = zoom_x1 - text_w - 5
            text_y = zoom_y1 + (zoomed_masked.height - text_h) // 2
            if text_x >= 0 and 0 <= text_y <= faded_rgba.height - text_h:
                draw.text((text_x, text_y), label_text,
                          fill="white", font=font)
                placed = True

        # Fallback: place the label anywhere it fits
        if not placed:
            for y in range(0, faded_rgba.height - text_h, 10):
                for x in range(0, faded_rgba.width - text_w, 10):
                    draw.text((x, y), label_text, fill="white", font=font)
                    placed = True
                    break
                if placed:
                    break

        # Resize back to original and return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faded_rgb = faded_rgba.convert("RGB")
        faded_rgb_np = np.array(faded_rgb)
        faded_rgb_resized = cv2.resize(
            faded_rgb_np, (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT), interpolation=cv2.INTER_AREA)
        faded_rgb_resized_pil = Image.fromarray(faded_rgb_resized)
        return faded_rgb_resized_pil

    def compress_gif(self, input_path, output_path, max_colors=64, resize_ratio=0.8):
        """
        Compresses a GIF by reducing the color palette and resizing each frame.

        Args:
            input_path (str or Path): Path to the input GIF file.
            output_path (str or Path): Path to save the compressed GIF.
            max_colors (int, optional): Maximum number of colors per frame. Defaults to 64.
            resize_ratio (float, optional): Resize factor for frame dimensions. Defaults to 0.8.

        Returns:
            None
        """
        with Image.open(input_path) as img:
            frames = []
            for frame in ImageSequence.Iterator(img):
                # Convert to optimized palette and resize
                frame = frame.convert(
                    'P', palette=Image.ADAPTIVE, colors=max_colors)
                frame = frame.resize(
                    (int(frame.width * resize_ratio),
                     int(frame.height * resize_ratio)),
                    Image.LANCZOS
                )
                frames.append(frame)

            # Save the optimized GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                loop=0,
                optimize=True
            )

    def createUI3(self, img_rgb, classes, confs, class_masks, out_dir):
        """
        Generates a GIF visualization of zoomed-in masked areas for the provided image and classes.

        Args:
            img_rgb (np.ndarray): The original RGB image as a NumPy array.
            classes (List[str]): List of class names to visualize.
            confs (Dict[str, float]): Confidence scores for each class.
            class_masks (Dict[str, np.ndarray]): Binary mask for each class.
            out_dir (str or Path): Directory to save output images and the final GIF.

        Returns:
            str: Path to the saved GIF file, or None if no frames were generated.
        """
        out_dir = Path(out_dir)
        all_frames_for_gif = []

        # Convert RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        for c in classes:
            if c in confs:
                zoomed_image = self.zoom_masked_area(
                    img_bgr.copy(),
                    class_masks[c].copy(),
                    c,
                    out_dir / f"zoomed_labeled_image_{c}.jpg"
                )

                # Convert zoomed image to NumPy array
                zoomed_image_np = np.array(zoomed_image)

                # Resize to original dimensions
                zoomed_image_resized = cv2.resize(
                    zoomed_image_np,
                    (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

                # Convert to PIL RGB format for GIF
                pil_img = Image.fromarray(cv2.cvtColor(
                    zoomed_image_resized, cv2.COLOR_BGR2RGB))
                all_frames_for_gif.append(pil_img)

        if not all_frames_for_gif:
            return None

        gif_path = out_dir / "ui3.gif"

        # Save animated GIF
        all_frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=all_frames_for_gif[1:],
            optimize=True,
            duration=1000,  # 1000ms per frame
            loop=0  # Infinite loop
        )

        return str(gif_path)

    def createUI5(self, img_rgb, classes, confs, class_masks, out_dir):
        """
        Generates a 1024x1024 resized GIF visualization of zoomed-in masked areas.

        Args:
            img_rgb (np.ndarray): The original RGB image as a NumPy array.
            classes (List[str]): List of class names to visualize.
            confs (Dict[str, float]): Confidence scores for each class.
            class_masks (Dict[str, np.ndarray]): Binary mask for each class.
            out_dir (str or Path): Directory to save output images and the final GIF.

        Returns:
            str: Path to the saved GIF file, or None if no frames were generated.
        """
        out_dir = Path(out_dir)
        all_frames_for_gif = []

        # Convert to BGR and resize to 1024x1024
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = np.array(Image.fromarray(img_bgr).resize(
            (1024, 1024), resample=Image.BILINEAR))

        for c in classes:
            if c in confs:
                zoomed_image = self.zoom_masked_area(
                    img_bgr.copy(),
                    class_masks[c].copy(),
                    c,
                    out_dir / f"zoomed_labeled_image_{c}.jpg"
                )

                zoomed_image_np = np.array(zoomed_image)

                # Resize to original resolution
                zoomed_image_resized = cv2.resize(
                    zoomed_image_np,
                    (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

                # Convert to PIL RGB format
                pil_img = Image.fromarray(cv2.cvtColor(
                    zoomed_image_resized, cv2.COLOR_BGR2RGB))
                all_frames_for_gif.append(pil_img)

        if not all_frames_for_gif:
            return None

        gif_path = out_dir / "ui5.gif"

        # Save animated GIF
        all_frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=all_frames_for_gif[1:],
            optimize=True,
            duration=1000,
            loop=0
        )

        return str(gif_path)

    def create_video_with_h264(self, frames, output_path: Path, fps=1):
        """
        Creates an H.264-encoded MP4 video from a list of RGB frames.

        Args:
            frames (List[np.ndarray]): List of RGB image frames.
            output_path (Path): Path where the final MP4 will be saved.
            fps (int): Frames per second for the output video.
        """
        temp_path = output_path.with_suffix(".temp.mp4")
        height, width, _ = frames[0].shape

        # Ensure dimensions are even (required by H.264)
        width -= width % 2
        height -= height % 2

        # Write raw MP4 using OpenCV with 'mp4v' codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (width, height))
            out.write(frame_resized)

        out.release()

        # First pass: Convert to H.264 compatible MP4
        final_output_temp = "temp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_path),
            "-vcodec", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", final_output_temp
        ], check=True)

        # Second pass: Final output to desired path
        subprocess.run([
            "ffmpeg", "-y", "-i", final_output_temp,
            "-vcodec", "libx264", "-profile:v", "high", str(output_path)
        ], check=True)

        temp_path.unlink()  # Clean up

    def createUI6(self, ori_img, img_rgb, classes, confs, class_masks, out_dir):
        """
        Generates an annotated MP4 video from original, processed, and zoomed class visualizations.

        Args:
            ori_img (np.ndarray): Original input image (BGR).
            img_rgb (np.ndarray): Processed image in RGB format.
            classes (List[str]): List of detected classes.
            confs (Dict[str, float]): Confidence scores for classes.
            class_masks (Dict[str, np.ndarray]): Mask for each class.
            out_dir (str or Path): Output directory to save the video.

        Returns:
            str: Path to the saved MP4 file.
        """
        out_dir = Path(out_dir)
        all_frames = []

        ori_img_copy = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_img_copy = cv2.resize(
            ori_img_copy, (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT))
        all_frames.append(ori_img_copy)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        all_frames.append(cv2.resize(
            img_bgr, (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT)))

        img_bgr = cv2.resize(img_rgb.copy(), (1024, 1024))

        for c in classes:
            if c not in confs:
                continue

            zoomed_image = self.zoom_masked_area(
                img_bgr.copy(), class_masks[c].copy(),
                c, out_dir / f"zoomed_labeled_image_{c}.jpg"
            )

            zoomed_resized = cv2.resize(
                np.array(zoomed_image),
                (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            all_frames.append(zoomed_resized)

        if not all_frames:
            return None

        height, width, _ = all_frames[0].shape
        temp_path = str(out_dir / "ui6_temp.mp4")
        final_path = str(out_dir / "ui6.mp4")

        # Write temporary MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 0.75, (width, height))

        for frame in all_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()

        # Convert to H.264 for better browser compatibility
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_path,
            "-vcodec", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", final_path
        ], check=True)

        return final_path

    def createUI2(self, overlay_path, classes, confs, class_masks, out_dir):
        """
        Places class labels onto an overlay image and saves it as a labeled PNG.

        Args:
            overlay_path (str or Path): Path to the overlay image (RGBA or RGB).
            classes (List[str]): List of class labels.
            confs (Dict[str, float]): Confidence scores per class.
            class_masks (Dict[str, np.ndarray]): Masks per class (same shape as image).
            out_dir (str or Path): Directory to save the final labeled image.

        Returns:
            Path: Path to the saved overlay image.
        """
        overlay_bgr2 = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)

        # Handle alpha channel (RGBA to RGB conversion)
        if overlay_bgr2.shape[2] == 4:
            b, g, r, a = cv2.split(overlay_bgr2)
            alpha = a.astype(float) / 255.0
            b = (b * alpha).astype(np.uint8)
            g = (g * alpha).astype(np.uint8)
            r = (r * alpha).astype(np.uint8)
            overlay_bgr2 = cv2.merge((b, g, r))

        img_height, img_width = overlay_bgr2.shape[:2]

        # Pad to 1024x1024 canvas
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        y_offset = max((1024 - img_height) // 2, 0)
        x_offset = max((1024 - img_width) // 2, 0)

        if img_height > 1024 or img_width > 1024:
            overlay_bgr2 = cv2.resize(
                overlay_bgr2, (min(img_width, 1024), min(img_height, 1024)),
                interpolation=cv2.INTER_AREA
            )
            img_height, img_width = overlay_bgr2.shape[:2]

        canvas[y_offset:y_offset + img_height,
               x_offset:x_offset + img_width] = overlay_bgr2
        overlay_bgr2 = canvas

        used_rects = []

        def overlaps(r1, r2):
            """Helper to detect rectangle overlap."""
            return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

        for c in classes:
            if c not in confs:
                continue

            mask = class_masks[c].astype(np.uint8)
            if cv2.countNonZero(mask) == 0:
                continue

            M = cv2.moments(mask)
            if M["m00"] == 0:
                continue

            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # Estimate local brightness
            bx, by = min(center_x, 1023), min(center_y, 1023)
            bgr_val = overlay_bgr2[by, bx]
            brightness = 0.299 * bgr_val[2] + \
                0.587 * bgr_val[1] + 0.114 * bgr_val[0]
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            text_to_put = self.clean_and_smart_title(c)
            font_scale = 1
            thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                text_to_put, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            text_x = max(0, min(center_x - text_width // 2, 1024 - text_width))
            text_y = max(text_height, min(
                center_y + text_height // 2, 1024 - baseline))

            rect = [text_x, text_y - text_height,
                    text_x + text_width, text_y + baseline]

            # Avoid overlaps with prior labels
            while any(overlaps(rect, prev) for prev in used_rects) and text_y + text_height < 1024:
                text_y += 10
                rect[1] += 10
                rect[3] += 10

            used_rects.append(rect)

            # Background enhancement for legibility
            pad = 5
            bg_x1, bg_y1 = max(
                text_x - pad, 0), max(text_y - text_height - pad, 0)
            bg_x2, bg_y2 = min(text_x + text_width + pad,
                               1024), min(text_y + baseline + pad, 1024)

            roi = overlay_bgr2[bg_y1:bg_y2, bg_x1:bg_x2].copy()
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)

            adjustment = 30 if text_color == (0, 0, 0) else -30
            adjusted = np.clip(blurred.astype(
                int) + adjustment, 0, 255).astype(np.uint8)

            blended = cv2.addWeighted(adjusted, 0.6, roi, 0.4, 0)
            overlay_bgr2[bg_y1:bg_y2, bg_x1:bg_x2] = blended

            # Draw the label
            cv2.putText(
                overlay_bgr2, text_to_put, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
            )

        # Final resize and save
        overlay_bgr2 = cv2.resize(
            overlay_bgr2, (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT),
            interpolation=cv2.INTER_AREA
        )
        labeled_path = out_dir / "overlay_labeled2.png"
        cv2.imwrite(str(labeled_path), overlay_bgr2)

        return labeled_path

    def createUI1(self, overlay_path, classes, confs, colors, out_dir):
        """
        Creates a labeled overlay image with class names and color-coded rectangles.

        Args:
            overlay_path (Path): Path to the base overlay image.
            classes (list): List of detected classes.
            confs (dict): Dictionary of confidence scores for detected classes.
            colors (dict): Mapping of class names to RGB color tuples.
            out_dir (Path): Directory to save the output image.

        Returns:
            Path: Path to the saved labeled overlay image.
        """
        overlay_bgr = cv2.imread(str(overlay_path))
        y = 40  # Initial vertical position for text labels

        for c in classes:
            if c in confs:
                text = f"{c}"
                # Convert color from RGB to BGR for OpenCV
                colors[c] = (colors[c][2], colors[c][1], colors[c][0])
                rect_color = colors[c]

                rect_x, rect_y = 10, y - 25
                rect_w, rect_h = 30, 30

                top_left = (rect_x, rect_y)
                bottom_right = (rect_x + rect_w, rect_y + rect_h)
                cv2.rectangle(overlay_bgr, top_left,
                              bottom_right, rect_color, -1)

                text_x = bottom_right[0] + 10
                cv2.putText(
                    overlay_bgr, text, (text_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA
                )
                y += 40  # Move down for the next label

        overlay_bgr = cv2.resize(
            overlay_bgr, (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        labeled_path = out_dir / "overlay_labeled.png"
        cv2.imwrite(str(labeled_path), overlay_bgr)
        return labeled_path

    def save_results(self, *, img_rgb, xyxy, names, class_masks, confs, classes, cmap, out_dir, original_image_path=None, generate_calories=False):
        """
        Processes segmentation/classification results, generates visualization overlays and masks.

        Args:
            img_rgb (np.ndarray): Original RGB image.
            xyxy (list): Bounding boxes (not used in current implementation).
            names (list): Class names.
            class_masks (dict): Dictionary mapping class names to binary masks.
            confs (dict): Dictionary of confidence scores for each class.
            classes (list): List of predicted classes.
            cmap (dict): Dictionary mapping class names to RGB color tuples.
            out_dir (Path): Directory to save the output visualizations.
            original_image_path (str, optional): Unused in this implementation.
            generate_calories (bool, optional): Unused in this implementation.

        Returns:
            Tuple[Path, list, dict, dict, dict, Path]: 
                - Path to saved overlay,
                - List of classes,
                - Confidence dictionary,
                - Class masks,
                - Color mapping,
                - Output directory
        """
        original_img = Image.fromarray(img_rgb)
        abs_mask = np.zeros_like(img_rgb)
        print("IN SAVE RESULTS:")
        print("TYPE OF CLASSES:", type(classes))
        print("TYPE OF CONFS:", type(confs))
        print("CONFS =", confs)

        colors = {}

        for c in classes:
            if c in confs:
                mask = class_masks[c]
                color = cmap.get(c)
                colors[c] = color

                if color is None:
                    self.logger.warning("No color for class %s", c)
                    continue
                abs_mask[mask] = color  # Apply color to absolute mask

        abs_mask = Image.fromarray(abs_mask)
        abs_mask.save(out_dir / "abs_mask.png")
        self.logger.info("Saved absolute mask for %s", out_dir.name)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(img_rgb).save(out_dir / "input.jpg")

        H, W = img_rgb.shape[:2]
        combined = np.zeros((H, W), bool)

        for c in classes:
            if c in confs:
                combined |= class_masks[c]

        Image.fromarray((combined * 255).astype('uint8')
                        ).save(out_dir / "combined_mask.png")
        self.logger.info("Saved combined mask for %s", out_dir.name)

        ov = img_rgb.copy()

        for c in classes:
            if c in confs:
                m = class_masks[c]
                col = cmap.get(c)
                ov[m] = ((ov[m].astype(float) * 0.3) +
                         np.array(col) * 0.7).astype('uint8')

        Image.fromarray(ov).save(out_dir / "overlay.png")
        self.logger.info("Saved classification overlay for %s", out_dir.name)

        overlay_path = out_dir / "overlay.png"

        return overlay_path, classes, confs, class_masks, colors, out_dir

    def remove_overlapping_masks(self,
                                 masks: np.ndarray,
                                 boxes: np.ndarray,
                                 names: list,
                                 containment_threshold: float = 0.9):
        """
        Removes masks that are mostly contained within larger masks, based on a containment threshold.

        Args:
            masks (np.ndarray): Boolean masks of shape (N, H, W).
            boxes (np.ndarray): Bounding boxes of shape (N, 4) in xyxy format.
            names (list): Class names associated with each mask.
            containment_threshold (float): Threshold to consider one mask contained in another (default is 0.9).

        Returns:
            tuple: (filtered_masks, filtered_boxes, filtered_names, removed_indices)
                - filtered_masks (np.ndarray): Masks after removal.
                - filtered_boxes (np.ndarray): Corresponding bounding boxes.
                - filtered_names (list): Corresponding class names.
                - removed_indices (list): Indices of removed masks.
        """
        N = masks.shape[0]
        keep = np.ones(N, dtype=bool)

        for i in range(N):
            if not keep[i]:
                continue
            for j in range(N):
                if i == j or not keep[j]:
                    continue

                inter = np.logical_and(masks[i], masks[j]).sum()
                area_i = masks[i].sum()
                containment = inter / (area_i + 1e-6)

                # If mask i is mostly inside mask j and j is larger, remove i
                if containment > containment_threshold and area_i < masks[j].sum():
                    keep[i] = False
                    break

        removed_indices = np.where(~keep)[0].tolist()
        return masks[keep], boxes[keep], [n for k, n in enumerate(names) if keep[k]], removed_indices

    def combine_all_masks(self, img_p, image_np, all_masks, kernel_size=9):
        """
        Combines all binary masks into one, applies morphological closing and hole filling,
        crops and resizes the region of interest, then removes background outside the mask.

        Args:
            img_p (str or Path): Output image path.
            image_np (np.ndarray): Original image in RGB or RGBA format.
            all_masks (list of np.ndarray): List of boolean masks.
            kernel_size (int): Kernel size for morphological closing (default: 9).

        Returns:
            PIL.Image.Image: Final resized and masked image.
        """
        image_rgba = Image.fromarray(image_np).convert("RGBA")
        image_np = np.array(image_rgba)

        # Combine all masks into a single union mask
        union_mask = all_masks[0]
        for e in all_masks:
            union_mask |= e

        # Morphological closing and filling small holes
        structure = np.ones((kernel_size, kernel_size), dtype=bool)
        closed_mask = binary_closing(union_mask, structure=structure)
        filled_mask = binary_fill_holes(closed_mask)

        # Get bounding box of the mask region
        ys, xs = np.where(filled_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        cropped_image = image_np.copy()
        cropped_mask = filled_mask

        # Resize cropped image and mask to original dimensions
        resized_image = np.array(Image.fromarray(cropped_image).resize(
            (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT), resample=Image.BILINEAR))
        resized_mask = np.array(Image.fromarray(cropped_mask.astype(np.uint8) * 255).resize(
            (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT), resample=Image.NEAREST)) > 0

        # Zero out regions outside the mask
        cropped_image[~cropped_mask] = [0, 0, 0, 0]
        resized_image[~resized_mask] = [0, 0, 0, 0]

        return Image.fromarray(resized_image)

    def save_cropped_image(self, img_p, masks, kernel_size=9,
                           output_path="/home/darpan/pipeline_code_darpan_local_copy/v12/cropped_dir",
                           cropped_image_name="final_segmented_image.png"):
        """
        Crops the image using a union of all masks, resizes it to the original dimensions,
        removes background pixels, and saves the final result.

        Args:
            img_p (str or Path): Path to the input image file.
            masks (List[np.ndarray]): List of boolean masks.
            kernel_size (int): Kernel size used for morphological closing (default: 9).
            output_path (str): Directory where cropped image is saved.
            cropped_image_name (str): Filename for the saved cropped image.

        Returns:
            str: Absolute path to the saved cropped image.
        """
        image = Image.open(img_p).convert("RGB")
        image = np.array(image)

        cropped_image = self.combine_all_masks(
            img_p, image, masks, kernel_size)

        final_path_to_save_cropped_image = os.path.join(
            output_path, cropped_image_name)
        Path(final_path_to_save_cropped_image).parent.mkdir(
            parents=True, exist_ok=True)

        cropped_image.save(final_path_to_save_cropped_image)
        return final_path_to_save_cropped_image

    def get_masks_bboxes_names(self, img_p):
        """
        Runs object detection and segmentation on the input image using Grounded SAM,
        then filters overlapping masks and returns final results.

        Args:
            img_p (str or Path): Path to the input image.

        Returns:
            tuple: (masks, boxes, class_names, confidences, removed_indices)
                - masks (np.ndarray): Boolean segmentation masks.
                - boxes (np.ndarray): Bounding boxes in xyxy format.
                - class_names (list): Class labels for each detected object.
                - confidences (np.ndarray): Detection confidence scores.
                - removed_indices (list): Indices of overlapping/removed masks.
        """
        self.logger.info("=== Processing %s ===", img_p.name)

        image = Image.open(img_p).convert("RGB")
        self.pred2.set_image(image)

        # Run Grounded DINO detection
        inputs = self.processor(
            images=image,
            text="food. liquid. plate. cup. bowl. cutlery.",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Post-process detections to extract boxes, scores, and labels
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.15,
            target_sizes=[image.size[::-1]]  # height x width
        )

        result = results[0]
        boxes = torch.tensor(result["boxes"]).cpu().numpy()
        confs_dino = torch.tensor(result["scores"]).cpu().numpy()
        names = result["labels"]

        self.logger.info("Obtained %d boxes from Grounded DINO", len(boxes))

        # Convert PIL image to numpy
        image = np.array(image)

        # Run SAM2 to get masks from boxes
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                masks, _, _ = self.pred2.predict(
                    None, None, box=boxes, multimask_output=False
                )

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.astype(bool)

        self.logger.info("Obtained %d masks from GSAM2", masks.shape[0])

        # Filter overlapping masks
        masks, boxes, names, removed_idx = self.remove_overlapping_masks(
            masks, boxes, names)

        return masks, boxes, names, confs_dino, removed_idx

    def names_of_food(self, img_p, masks, xyxy, names, confs_dino):
        """
        Classifies the segmented food items from the image using a prototype embedding-based classifier,
        and saves classification overlays and metadata for matched menu items.

        Args:
            img_p (Path): Path to the input image file.
            masks (np.ndarray): Boolean segmentation masks.
            xyxy (np.ndarray): Bounding boxes for each mask.
            names (list): Class names from initial detection.
            confs_dino (np.ndarray): Confidence scores from object detection.

        Returns:
            tuple: (
                overlay_path (Path),
                classes (List[str]),
                confs (Dict[str, float]),
                class_masks (Dict[str, np.ndarray]),
                colors (Dict[str, Tuple[int, int, int]]),
                out_dir (Path),
                img_rgb (np.ndarray)
            )
        """
        with open(self.args.menu_json) as f:
            date_to_menu = json.load(f)

        # Get today's menu
        menu = date_to_menu.get(self.curdate)
        if menu is None:
            self.logger.warning(
                "No menu for date %s, skipping %s", self.curdate, img_p.name)
            return None, None, None, None, None, None, None

        self.logger.info("Today's menu (%s): %s", self.curdate, menu)

        out = Path(self.args.output_dir) / img_p.stem
        out.mkdir(parents=True, exist_ok=True)

        # Read RGB image using OpenCV
        img_rgb = cv2.imread(str(img_p))[..., ::-1]

        # Load prototype embeddings for current date
        proto_embs = self.load_prototype_embeddings(
            "/home/darpan/pipeline_code_darpan_local_copy/embeddings_PE-Core-L14-336_selcted_30", self.curdate)

        # Load experiment config and initialize classifier
        cfg = ExperimentConfig.load(self.args.config)
        classif = PEClassifier(
            self.pe_model,
            self.preprocess,
            cfg.similarity.threshold,
            self.device,
            num_patches=cfg.similarity.num_patches,
            patch_size=cfg.similarity.patch_size,
            sim_method=cfg.similarity.type,
            knn_k=cfg.similarity.knn_k
        )

        # Classify each mask using the PEClassifier
        class_masks, class_confs = classif.classify(
            masks.copy(), img_rgb.copy(), proto_embs)
        self.logger.info("Classification complete: %s",
                         list(class_confs.keys()))

        # Filter predictions by today's menu
        filtered_masks = {c: class_masks[c] for c in menu if c in class_masks}
        filtered_confs = {c: class_confs[c] for c in menu if c in class_confs}

        # Load colormap and save visual + textual results
        classes_all, class2rgb = self.load_colormap(self.args.colormap_csv)

        overlay_path, classes, confs, class_masks, colors, out_dir = self.save_results(
            img_rgb=img_rgb.copy(),
            xyxy=xyxy.copy(),
            names=names,
            class_masks=filtered_masks.copy(),
            confs=filtered_confs.copy(),
            classes=menu.copy(),
            cmap=class2rgb.copy(),
            out_dir=out,
            original_image_path=img_p,
            generate_calories=False
        )

        return overlay_path, classes, confs, class_masks, colors, out_dir, img_rgb.copy()

    def resize_image_in_place(self, image_path, new_size):
        """
        Resizes an image in-place to the given size. The original is renamed temporarily for undo.

        Args:
            image_path (Path): Path to the image file to resize.
            new_size (tuple): New size as (width, height).
        """
        image_path_str = str(image_path)
        img = cv2.imread(image_path_str)

        height, width = img.shape[:2]
        if (width, height) != new_size:
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            temp_path = f"{str(image_path.parent)}/{image_path.stem}_temp{image_path.suffix}"
            os.rename(image_path_str, temp_path)
            cv2.imwrite(image_path_str, resized)

    def undo_resizing(self, image_path):
        """
        Reverts the image back to its original size using the temporary backup created during resizing.

        Args:
            image_path (str): Path to the resized image.
        """
        img = cv2.imread(image_path)
        if (self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT) != (img.shape[1], img.shape[0]):
            if os.path.exists(image_path):
                os.remove(image_path)
            temp_path = f"{str(image_path.parent)}/{image_path.stem}_temp{image_path.suffix}"
            os.rename(temp_path, str(image_path))

    def main_fn(self):
        """
        Main execution function to run the food segmentation and classification pipeline on one
        or more input images. The pipeline includes:

        1. Reading and resizing the image.
        2. Detecting food-related objects and generating masks.
        3. Saving cropped/segmented image.
        4. Filtering masks to only food-related items.
        5. Classifying masks using prototype embeddings.
        6. Saving visual overlays and launching the final UI.
        7. Undoing image resize to restore original image.

        Raises:
            ValueError: If any image cannot be read.
        """
        # Collect all image paths based on input arguments
        img_paths = ([Path(self.args.image)] if self.args.image
                     else sorted(Path(self.args.img_dir).glob('*.*')))

        for img_p in img_paths:
            curimg = cv2.imread(img_p)
            if curimg is None:
                raise ValueError(f"Could not read image at {img_p}")

            # Store original image dimensions
            self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH = curimg.shape[:2]

            # Resize image in place to standard 1024x1024 for consistent processing
            self.resize_image_in_place(img_p, (1024, 1024))

            # Step 1: Detect masks, bounding boxes, and labels using SAM + Grounding DINO
            masks, xyxy, names, confs_dino, removed_idxs = self.get_masks_bboxes_names(
                img_p)

            # Remove detections that were filtered out earlier
            confs_dino = np.delete(confs_dino, removed_idxs, axis=0)

            # Step 2: Save combined cropped image from all masks
            new_pth = self.save_cropped_image(
                img_p, masks.copy(), kernel_size=9)
            self.new_pth = Path(new_pth)

            # Step 3: Filter masks to only those labeled as "food"
            valid_indices = [i for i, label in enumerate(
                names) if "food" in label]
            masks = masks[valid_indices]
            xyxy = xyxy[valid_indices]
            names = [names[i] for i in valid_indices]
            confs_dino = confs_dino[valid_indices]

            # Step 4: Classify the segmented food items using prototype embeddings
            overlay_path, classes, confs, class_masks, colors, out_dir, img_rgb = self.names_of_food(
                img_p, masks.copy(), xyxy.copy(), names, confs_dino
            )

            # Step 5: Load the cropped image and render final UI visualization
            ci = cv2.imread(self.new_pth)
            self.createUI6(
                ori_img=curimg,
                img_rgb=ci,
                classes=classes,
                confs=confs,
                class_masks=class_masks,
                out_dir=out_dir
            )

            # Step 6: Revert resized image back to its original dimensions
            self.undo_resizing(img_p)


# for testing purposes
if __name__ == "__main__":
    proj = FoodSeg()
    proj.main_fn()
