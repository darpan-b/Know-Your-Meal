import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms
from pdb import set_trace as stx

# --- Configuration ---
PE_CONFIG = "PE-Core-L14-336"
# PE_CONFIG       = "PE-Spatial-G14-448"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Flag to determine whether to save mean embeddings or all patch embeddings
use_mean = False  # Set to True for saving class-level mean, False for saving all patches

# Set output directory based on the use_mean flag
if use_mean:
    EMBEDDINGS_DIR = f"embeddings_{PE_CONFIG}_selcted_30_mean"
else:
    # EMBEDDINGS_DIR = "/home/darpan/summer-school-embeddings"
    EMBEDDINGS_DIR = "/home/darpan/temp_prototype_embeddings_patches"
    # EMBEDDINGS_DIR = f"embeddings_{PE_CONFIG}_selcted_30_patches"

CENTER_CROP_SIZE = 224
OFFSET_PERCENTAGE = 0.1  # Adjust as needed for the offset crops


def load_pe_model(config: str = PE_CONFIG):
    """
    Loads the PE‐Core model and its preprocess transform.
    Returns: (model, preprocess_fn)
    """
    if config == 'PE-Spatial-G14-448':
        model = pe.VisionTransformer.from_config(
            config, pretrained=True).to(DEVICE).eval()  # Loads from HF
    else:
        # Load the PE-Core model
        model = pe.CLIP.from_config(config, pretrained=True).to(DEVICE).eval()
    preprocess = pe_transforms.get_image_transform(model.image_size)
    return model, preprocess


def preprocess_image(preprocess_fn, image: Image.Image) -> torch.Tensor:
    """Preprocesses a PIL Image to a tensor."""
    return preprocess_fn(image).unsqueeze(0).to(DEVICE)


def encode_image(model, img_tensor: torch.Tensor) -> np.ndarray:
    """Runs the vision encoder and returns a normalized 1×D numpy vector."""

    with torch.no_grad(), torch.autocast("cuda"):
        try:
            feat = model.encode_image(img_tensor)           # (1, D)
        except AttributeError:  # Fallback if encode_image is not available
            try:
                feat = model.forward_features(img_tensor)           # (1, D)
            except Exception as e:
                print(f"Error in model encoding: {e}")
                return None

    # Ensure feat is a tensor before converting to numpy
    if isinstance(feat, torch.Tensor):
        arr = feat.cpu().numpy().reshape(-1)
        # Normalization will be applied either per patch (if !use_mean)
        # or at the end for the final class mean (if use_mean)
        return arr
    else:
        print("Model output was not a tensor.")
        return None


def save_embeddings(emb: np.ndarray, save_path: str, normalize: bool = True):
    """Saves a 1-D embedding to .npy, creating dirs as needed, with optional normalization."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if normalize:
        emb /= (np.linalg.norm(emb) + 1e-8)
    np.save(save_path, emb)


def get_center_crops(image: Image.Image, crop_size: int, offset_percentage: float = 0.1):
    """Generates 5 center crops: 1 absolute center and 4 slightly offset."""
    width, height = image.size

    center_x = width // 2
    center_y = height // 2

    half_crop = crop_size // 2

    crops = []
    bboxes = []  # Store bboxes for potential debugging or validation

    # 1. Absolute center crop
    bbox_center = (center_x - half_crop, center_y - half_crop,
                   center_x + half_crop, center_y + half_crop)
    crops.append(image.crop(bbox_center))
    bboxes.append(bbox_center)

    # Calculate offset
    x_offset = int(width * offset_percentage) // 2
    y_offset = int(height * offset_percentage) // 2

    # Define offset bboxes
    bbox_left = (center_x - half_crop - x_offset, center_y - half_crop,
                 center_x + half_crop - x_offset, center_y + half_crop)
    bbox_right = (center_x - half_crop + x_offset, center_y - half_crop,
                  center_x + half_crop + x_offset, center_y + half_crop)
    bbox_top = (center_x - half_crop, center_y - half_crop - y_offset,
                center_x + half_crop, center_y + half_crop - y_offset)
    bbox_bottom = (center_x - half_crop, center_y - half_crop +
                   y_offset, center_x + half_crop, center_y + half_crop + y_offset)

    offset_bboxes = [bbox_left, bbox_right, bbox_top, bbox_bottom]

    # Add offset crops only if they are within image bounds
    for i, bbox in enumerate(offset_bboxes):
        # Check if bounding box is within image dimensions (x1, y1, x2, y2)
        if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= width and bbox[3] <= height:
            crops.append(image.crop(bbox))
            bboxes.append(bbox)
        else:
            # Optional: Log if a crop is skipped
            # direction = ["Left", "Right", "Top", "Bottom"][i]
            # tqdm.write(f"    Skipping {direction} crop, out of bounds: {bbox} for image size ({width}, {height}).")
            pass  # Silently skip out-of-bounds crops

    # Note: The number of crops returned can be less than 5 if offsets go out of bounds

    return crops  # , bboxes # Can return bboxes as well if needed


def get_center_crops_and_save(image: Image.Image, crop_size: int, save_dir: str, image_id: str = "image", offset_percentage: float = 0.1):
    """
    Generates and saves up to 5 center crops (1 center + 4 offsets) from the image.

    Args:
        image (PIL.Image): Input image.
        crop_size (int): Size of the square crops.
        save_dir (str): Directory where crops will be saved.
        image_id (str): Prefix for saved filenames (e.g., original filename without extension).
        offset_percentage (float): Offset as a percentage of image size.

    Returns:
        List[PIL.Image]: List of cropped image objects.
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    half_crop = crop_size // 2

    crops = []
    directions = ['center', 'left', 'right', 'top', 'bottom']

    # 1. Absolute center crop
    bbox_center = (center_x - half_crop, center_y - half_crop,
                   center_x + half_crop, center_y + half_crop)
    crops.append(image.crop(bbox_center))

    # Save the center crop
    crops[-1].save(os.path.join(save_dir, f"{image_id}_center.jpg"))

    # Calculate offset
    x_offset = int(width * offset_percentage) // 2
    y_offset = int(height * offset_percentage) // 2

    # Define offset bboxes
    offset_bboxes = [
        (center_x - half_crop - x_offset, center_y - half_crop,
         center_x + half_crop - x_offset, center_y + half_crop),  # left
        (center_x - half_crop + x_offset, center_y - half_crop,
         center_x + half_crop + x_offset, center_y + half_crop),  # right
        (center_x - half_crop, center_y - half_crop - y_offset,
         center_x + half_crop, center_y + half_crop - y_offset),  # top
        (center_x - half_crop, center_y - half_crop + y_offset,
         center_x + half_crop, center_y + half_crop + y_offset),  # bottom
    ]

    # Loop through offset crops
    for i, bbox in enumerate(offset_bboxes):
        if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= width and bbox[3] <= height:
            crop = image.crop(bbox)
            crops.append(crop)
            crop.save(os.path.join(
                save_dir, f"{image_id}_{directions[i+1]}.jpg"))

    return crops


def precompute_daily_embeddings(
    prototype_dir: str,
    embeddings_root: str = EMBEDDINGS_DIR,
    # patches_dir: str = None,
    use_mean: bool = False  # Pass the flag here too
):
    model, preprocess = load_pe_model()

    date_folders = [
        d for d in os.listdir(prototype_dir)
        if os.path.isdir(os.path.join(prototype_dir, d)) and d.isdigit()
    ]

    for date_str in tqdm(sorted(date_folders), desc="Processing Dates"):
        date_folder_path = os.path.join(prototype_dir, date_str)
        # Output directory for this date
        date_out_dir = os.path.join(embeddings_root, date_str)
        os.makedirs(date_out_dir, exist_ok=True)
        print(f"[{date_str}] processing")

        class_folders = [
            d for d in os.listdir(date_folder_path)
            if os.path.isdir(os.path.join(date_folder_path, d))
        ]

        for cls in tqdm(sorted(class_folders), desc="Processing Classes", leave=False):
            class_folder = os.path.join(date_folder_path, cls)

            # Output directory for this class (only needed if saving patches)
            class_out_dir = os.path.join(date_out_dir, cls)
            if not use_mean:
                os.makedirs(class_out_dir, exist_ok=True)

            image_files = [
                os.path.join(class_folder, fn) for fn in os.listdir(class_folder)
                if fn.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            # List to store mean embeddings for each image in the class (only if use_mean)
            all_image_mean_embeddings = []
            processed_image_count = 0  # Counter for images successfully processed

            for img_path in tqdm(sorted(image_files), desc=f"images→{cls}", leave=False):
                try:
                    img = Image.open(img_path).convert("RGB")
                    # crops = get_center_crops_and_save(
                    #     img, CENTER_CROP_SIZE, save_dir=patches_dir, offset_percentage=OFFSET_PERCENTAGE)
                    crops = get_center_crops(
                        img, CENTER_CROP_SIZE, OFFSET_PERCENTAGE)
                    image_basename = os.path.splitext(
                        os.path.basename(img_path))[0]

                    image_embeddings = []  # List to store embeddings for the patches of this image
                    for i, crop in enumerate(crops):
                        inp = preprocess_image(preprocess, crop)
                        patch_embedding = encode_image(model, inp)

                        if patch_embedding is not None:
                            image_embeddings.append(patch_embedding)
                        else:
                            tqdm.write(
                                f"    ⚠️ Skipping patch {i} for {os.path.basename(img_path)} due to encoding error.")

                    if image_embeddings:
                        if use_mean:
                            # Calculate mean embedding for this image
                            mean_image_embedding = np.stack(
                                image_embeddings, axis=0).mean(axis=0)
                            all_image_mean_embeddings.append(
                                mean_image_embedding)
                            processed_image_count += 1  # Count image if we successfully got embeddings for it

                        else:  # Save individual patch embeddings
                            for i, patch_emb in enumerate(image_embeddings):
                                # Define save path for the individual patch embedding
                                save_path = os.path.join(
                                    class_out_dir, f"{image_basename}_patch_{i}.npy")
                                # Normalize each patch embedding before saving
                                save_embeddings(
                                    patch_emb, save_path, normalize=True)
                            processed_image_count += 1  # Count image if we successfully saved patches

                except Exception as e:
                    tqdm.write(
                        f"    ⚠️ Error processing {os.path.basename(img_path)}: {e}")

            # After processing all images in a class
            if use_mean:
                if all_image_mean_embeddings:
                    # Calculate the mean of the mean embeddings for the class
                    mean_class_embedding = np.stack(
                        all_image_mean_embeddings, axis=0).mean(axis=0)
                    # Save the class-level mean embedding (normalize it)
                    save_path = os.path.join(date_out_dir, f"{cls}.npy")
                    save_embeddings(mean_class_embedding,
                                    save_path, normalize=True)
                else:
                    tqdm.write(
                        f"    ⚠️ No valid image embeddings found for class {cls} in {date_str} to compute mean.")
            else:
                if processed_image_count == 0:
                    tqdm.write(
                        f"    ⚠️ No patches saved for class {cls} in {date_str}.")


if __name__ == "__main__":
    # "/home/nutrition/code/yash/food-classify/selected_30/prototypes_cutout_masks_resized_30"
    PROTO_DIR = "/home/darpan/temp_prototype_images"
    # PATCHES_DIR = "/home/darpan/temp_prototype_patches"
    # Pass the flag to the function
    # precompute_daily_embeddings(PROTO_DIR, PATCHES_DIR, use_mean=use_mean)
    precompute_daily_embeddings(PROTO_DIR, use_mean=use_mean)
