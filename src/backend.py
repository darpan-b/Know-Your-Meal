from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import base64
import uuid
import cv2
import numpy as np
import mimetypes
import time
from urllib.parse import quote
import psutil
import os
import logging
from datetime import datetime
from functools import wraps
import tracemalloc
import gc

from pipeline_v12_aa import FoodSeg

# --- Memory Profiling Setup ---


class MemoryProfiler:
    def __init__(self, log_file="memory_profile.log"):
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration for memory profiling"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger("MemoryProfiler")

    def get_memory_info(self):
        """Get current memory usage information"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "memory_percent": memory_percent,
            "available_memory_mb": virtual_memory.available / 1024 / 1024,
            "total_memory_mb": virtual_memory.total / 1024 / 1024,
            "memory_usage_percent": virtual_memory.percent
        }

    def log_memory_usage(self, operation_name, session_id=None):
        """Log current memory usage with operation context"""
        memory_info = self.get_memory_info()
        session_info = f" [Session: {session_id}]" if session_id else ""

        self.logger.info(
            f"{operation_name}{session_info} - "
            f"RSS: {memory_info['rss_mb']:.2f}MB, "
            f"VMS: {memory_info['vms_mb']:.2f}MB, "
            f"Process Memory: {memory_info['memory_percent']:.2f}%, "
            f"System Memory: {memory_info['memory_usage_percent']:.2f}%, "
            f"Available: {memory_info['available_memory_mb']:.2f}MB"
        )

    def profile_function(self, operation_name):
        """Decorator to profile memory usage of a function"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Start tracemalloc for detailed memory tracking
                tracemalloc.start()

                # Log memory before operation
                self.log_memory_usage(f"{operation_name} - START")
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Log memory after successful operation
                    end_time = time.time()
                    duration = end_time - start_time

                    # Get tracemalloc snapshot
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    self.log_memory_usage(f"{operation_name} - SUCCESS")
                    self.logger.info(
                        f"{operation_name} - Duration: {duration:.2f}s, "
                        f"Current traced memory: {current / 1024 / 1024:.2f}MB, "
                        f"Peak traced memory: {peak / 1024 / 1024:.2f}MB"
                    )

                    # Force garbage collection and log memory after cleanup
                    gc.collect()
                    self.log_memory_usage(f"{operation_name} - AFTER_GC")

                    return result

                except Exception as e:
                    # Log memory after failed operation
                    end_time = time.time()
                    duration = end_time - start_time

                    if tracemalloc.is_tracing():
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        self.logger.error(
                            f"{operation_name} - ERROR after {duration:.2f}s: {str(e)}, "
                            f"Current traced memory: {current / 1024 / 1024:.2f}MB, "
                            f"Peak traced memory: {peak / 1024 / 1024:.2f}MB"
                        )
                    else:
                        self.logger.error(
                            f"{operation_name} - ERROR after {duration:.2f}s: {str(e)}")

                    self.log_memory_usage(f"{operation_name} - ERROR")
                    raise

            return wrapper
        return decorator


# Initialize memory profiler
memory_profiler = MemoryProfiler("foodseg_memory_profile.log")

# --- Initialize FastAPI app ---
app = FastAPI(title="FoodSeg FastAPI Modular App")

# --- Constants ---
UPLOAD_DIR = Path("uploaded_images")
OUTPUT_DIR = Path(
    "/home/darpan/pipeline_code_darpan_local_copy/v12/output_dir")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
mimetypes.add_type('video/mp4', '.mp4')

# --- Serve static media with correct MIME types ---
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


@app.get("/static/{file_path:path}")
async def serve_static_file(file_path: str):
    full_path = OUTPUT_DIR / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if full_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500, detail="File is empty or corrupted")

    mime_type, _ = mimetypes.guess_type(str(full_path))
    try:
        return FileResponse(full_path, media_type=mime_type or "application/octet-stream")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to serve file: {str(e)}")


# --- Enable CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session storage ---
SESSIONS = {}
fs = None

# --- Run once on startup ---


@app.on_event("startup")
def init_model():
    global fs
    memory_profiler.log_memory_usage(
        "APPLICATION_STARTUP - Before FoodSeg init")
    fs = FoodSeg()
    memory_profiler.log_memory_usage(
        "APPLICATION_STARTUP - After FoodSeg init")
    print("✅ FoodSeg initialized.")

# --- Pydantic model ---


class UploadPayload(BaseModel):
    image: str  # base64 PNG
    date: str

# --- Helper function to determine media type ---


def get_media_type(file_path):
    """Determine if the file is an image or video based on extension"""
    file_extension = Path(file_path).suffix.lower()

    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.gif', '.bmp', '.webp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov',
                        '.wmv', '.flv', '.webm', '.mkv'}

    if file_extension in image_extensions:
        return "image"
    elif file_extension in video_extensions:
        return "video"
    else:
        return "unknown"

# --- Upload and process image ---


@app.post("/upload")
@memory_profiler.profile_function("IMAGE_UPLOAD")
async def upload_image(payload: UploadPayload):
    session_id = None
    try:
        global fs
        if not fs:
            raise HTTPException(
                status_code=500, detail="FoodSeg not initialized")

        try:
            memory_profiler.log_memory_usage(
                "IMAGE_UPLOAD - Base64 decode start")
            image_data = base64.b64decode(payload.image)
            session_id = str(uuid.uuid4())
            img_path = UPLOAD_DIR / f"{session_id}.png"
            with open(img_path, "wb") as f:
                f.write(image_data)
            memory_profiler.log_memory_usage(
                "IMAGE_UPLOAD - Image saved", session_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        cd = payload.date  # YYYYMMDD
        print("cd is", payload.date)
        fs.curdate = cd.replace("-", "")
        print("CURRENT DATE IS", fs.curdate)

        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Before cv2.imread", session_id)
        cur_img = cv2.imread(str(img_path))
        if cur_img is None:
            raise HTTPException(
                status_code=400, detail="Invalid image content")

        fs.ORIGINAL_HEIGHT, fs.ORIGINAL_WIDTH = cur_img.shape[:2]
        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Before resize", session_id)
        fs.resize_image_in_place(img_path, (1024, 1024))
        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - After resize", session_id)

        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Before get_masks_bboxes_names", session_id)
        masks, xyxy, names, confs, removed, raw_masks_dir = fs.get_masks_bboxes_names(
            img_path, out_dir=OUTPUT_DIR)

        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - After get_masks_bboxes_names", session_id)

        confs = np.delete(confs, removed, axis=0)
        cropped_path1 = OUTPUT_DIR / session_id
        cropped_path1.mkdir(parents=True, exist_ok=True)
        print("cropped path 1 is", cropped_path1)

        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Before save_cropped_image", session_id)
        cropped_path = fs.save_cropped_image(img_p=img_path, masks=masks.copy(),
                                             kernel_size=9, output_path=cropped_path1)
        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - After save_cropped_image", session_id)

        fs.new_pth = Path(cropped_path)
        print("fs.new_pth =", fs.new_pth, " OUTPUT DIR =", OUTPUT_DIR)
        rel_path = fs.new_pth.relative_to(OUTPUT_DIR)
        cropped_image_url = f"http://drishti.iiit.ac.in:8002/static/{quote(str(rel_path))}"

        valid = [i for i, name in enumerate(names) if "food" in name]
        masks, xyxy, names, confs = masks[valid], [
            xyxy[i] for i in valid], [names[i] for i in valid], confs[valid]

        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Before names_of_food", session_id)
        overlay_path, classes, confidences, class_masks, colors, out_dir, img_rgb, groundingdino_path, grounded_sam_path, gsam_json_log_path = fs.names_of_food(
            img_path, masks.copy(), xyxy.copy(), names, confs
        )
        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - After names_of_food", session_id)

        SESSIONS[session_id] = {
            "img_path": str(img_path),
            "overlay_path": str(overlay_path),
            "new_pth": str(cropped_path),
            "cropped_path_url": cropped_image_url,
            "classes": classes,
            "confs": confidences,
            "colors": colors,
            "out_dir": str(out_dir),
            "class_masks": class_masks,
            "img_rgb": img_rgb,
            "groundingdino_path": str(groundingdino_path),
            "grounded_sam_path": str(grounded_sam_path),
            "gsam_json_log_path": str(gsam_json_log_path),
            "raw_masks_dir": str(raw_masks_dir)
        }

        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Session stored", session_id)
        memory_profiler.logger.info(
            f"IMAGE_UPLOAD - Session {session_id} completed successfully. Active sessions: {len(SESSIONS)}")

        return JSONResponse({
            "session_id": session_id,
            "overlay_image": str(overlay_path),
            "cropped_output": str(cropped_path),
            "classes_detected": classes,
            "confidences": confidences,
            "cropped_path_url": cropped_image_url,
            "out_dir": str(out_dir),
            "ui_modes": ["ui1", "ui2", "ui3", "ui4", "ui6"],
            "groundingdino_path": str(groundingdino_path),
            "grounded_sam_path": str(grounded_sam_path),
            "gsam_json_log_path": str(gsam_json_log_path),
            "raw_masks_dir": str(raw_masks_dir)
        })
    except Exception as e:
        memory_profiler.log_memory_usage(
            "IMAGE_UPLOAD - Exception occurred", session_id)
        raise HTTPException(
            status_code=500, detail=f"Image upload error: {str(e)}")

# --- Generate UI asset ---


class GeneratePayload(BaseModel):
    ui_mode: str
    session_id: str


@app.post("/create")
@memory_profiler.profile_function("UI_CREATION")
async def create_ui(payload: GeneratePayload):
    session_id = payload.session_id
    ui_mode = payload.ui_mode

    print("SESSION ID IS", session_id)
    print("UI MODE IS", ui_mode)

    memory_profiler.log_memory_usage(
        f"UI_CREATION - {ui_mode.upper()} start", session_id)

    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    media_path, individual_image_paths = None, None

    try:
        print("mode =", ui_mode)
        if ui_mode == "ui1":
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI1 before createUI1", session_id)
            media_path = fs.createUI1(
                overlay_path=session["overlay_path"],
                classes=session["classes"],
                confs=session["confs"],
                colors=session["colors"],
                out_dir=Path(session["out_dir"])
            )
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI1 after createUI1", session_id)

        elif ui_mode == "ui2":
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI2 before createUI2", session_id)
            media_path = fs.createUI2(
                overlay_path=session["new_pth"],
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI2 after createUI2", session_id)

        elif ui_mode == "ui3":
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI3 before createUI3", session_id)
            media_path = fs.createUI3(
                img_rgb=session["img_rgb"].copy(),
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI3 after createUI3", session_id)

        elif ui_mode == "ui4":
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI4 before image load", session_id)
            img_rgb_cur = cv2.imread(fs.new_pth)
            print(type(img_rgb_cur))
            print("HEY, IMAGE IS PRESENT! and its dimensions are", img_rgb_cur.shape)
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI4 before createUI5", session_id)
            media_path = fs.createUI5(
                img_rgb=img_rgb_cur.copy(),
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI4 after createUI5", session_id)

        elif ui_mode == "ui6":
            print("this start")
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI6 before image loads", session_id)
            ori_img = cv2.imread(session["img_path"])
            img_rgb_cur = cv2.imread(fs.new_pth)
            print(type(img_rgb_cur))
            print("HEY, IMAGE IS PRESENT! and its dimensions are", img_rgb_cur.shape)
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI6 before createUI6", session_id)
            media_path, individual_image_paths = fs.createUI6(
                ori_img=ori_img.copy(),
                img_rgb=img_rgb_cur.copy(),
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )
            memory_profiler.log_memory_usage(
                "UI_CREATION - UI6 after createUI6", session_id)
            print("this done")

        else:
            raise HTTPException(status_code=400, detail="Invalid UI mode")

        if not media_path:
            raise HTTPException(status_code=500, detail="UI generation failed")

        # Determine media type
        media_type = get_media_type(media_path)

        rel_path = Path(media_path).relative_to(OUTPUT_DIR)
        print("rel path is", rel_path)
        media_url = f"http://drishti.iiit.ac.in:8002/static/{rel_path}"

        memory_profiler.log_memory_usage(
            f"UI_CREATION - {ui_mode.upper()} completed successfully", session_id)
        memory_profiler.logger.info(
            f"UI_CREATION - {ui_mode.upper()} for session {session_id} completed. Media path: {media_path}")

        if individual_image_paths:
            individual_image_paths = [str(p) for p in individual_image_paths]

        return JSONResponse({
            "message": f"{ui_mode.upper()} created successfully.",
            "media_path": media_url,
            "media_type": media_type,
            "file_extension": str(Path(media_path).suffix.lower()),
            "individual_image_paths": individual_image_paths or []
        })

    except Exception as e:
        memory_profiler.log_memory_usage(
            f"UI_CREATION - {ui_mode.upper()} error", session_id)
        raise HTTPException(
            status_code=500, detail=f"UI creation error: {str(e)}")

# Add a cleanup endpoint for memory management (optional)


@app.post("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Optional endpoint to clean up session data and free memory"""
    memory_profiler.log_memory_usage(
        "CLEANUP - Before session cleanup", session_id)

    if session_id in SESSIONS:
        del SESSIONS[session_id]
        gc.collect()  # Force garbage collection
        memory_profiler.log_memory_usage(
            "CLEANUP - After session cleanup", session_id)
        memory_profiler.logger.info(
            f"CLEANUP - Session {session_id} cleaned up. Active sessions: {len(SESSIONS)}")
        return JSONResponse({"message": f"Session {session_id} cleaned up successfully"})
    else:
        raise HTTPException(status_code=404, detail="Session not found")
