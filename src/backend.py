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

from pipeline_v12 import FoodSeg

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


# @app.get("/static/{file_path:path}")
# async def servse_static_file(file_path: str):
#     full_path = OUTPUT_DIR / file_path
#     if not full_path.exists():
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(full_path)


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
start_time = time.time()  # Track the start time for re-initialization
# --- Run once on startup ---


@app.on_event("startup")
def init_model():
    global fs, start_time
    start_time = time.time()  # Reset start time on startup
    fs = FoodSeg()
    print("✅ FoodSeg initialized.")

# --- Pydantic model ---


def reinitialize_fs():
    global fs
    fs = FoodSeg()
    print("✅ FoodSeg re-initialized.")


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
async def upload_image(payload: UploadPayload):
    try:
        global fs
        current_time = time.time()
        if current_time - start_time > 3600*5:  # 5 hours
            print("Re-initializing FoodSeg due to timeout")
            reinitialize_fs()
            start_time = current_time
        if not fs:
            raise HTTPException(
                status_code=500, detail="FoodSeg not initialized")

        try:
            image_data = base64.b64decode(payload.image)
            session_id = str(uuid.uuid4())
            img_path = UPLOAD_DIR / f"{session_id}.png"
            with open(img_path, "wb") as f:
                f.write(image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        cd = payload.date  # YYYYMMDD
        print("cd is", payload.date)
        fs.curdate = cd.replace("-", "")
        print("CURRENT DATE IS", fs.curdate)
        cur_img = cv2.imread(str(img_path))
        if cur_img is None:
            raise HTTPException(
                status_code=400, detail="Invalid image content")

        fs.ORIGINAL_HEIGHT, fs.ORIGINAL_WIDTH = cur_img.shape[:2]
        fs.resize_image_in_place(img_path, (1024, 1024))

        masks, xyxy, names, confs, removed = fs.get_masks_bboxes_names(
            img_path)
        confs = np.delete(confs, removed, axis=0)
        cropped_path1 = OUTPUT_DIR / session_id
        cropped_path1.mkdir(parents=True, exist_ok=True)
        print("cropped path 1 is", cropped_path1)
        cropped_path = fs.save_cropped_image(img_p=img_path, masks=masks.copy(
        ), kernel_size=9, output_path=cropped_path1)
        fs.new_pth = Path(cropped_path)
        print("fs.new_pth =", fs.new_pth, " OUTPUT DIR =", OUTPUT_DIR)
        rel_path = fs.new_pth.relative_to(OUTPUT_DIR)
        cropped_image_url = f"http://drishti.iiit.ac.in:8002/static/{quote(str(rel_path))}"
        # cropped_image = cv2.imread(fs.new_pth)
        # fs.ORIGINAL_HEIGHT, fs.ORIGINAL_WIDTH = cropped_image.shape[:2]

        valid = [i for i, name in enumerate(names) if "food" in name]
        masks, xyxy, names, confs = masks[valid], [
            xyxy[i] for i in valid], [names[i] for i in valid], confs[valid]

        overlay_path, classes, confidences, class_masks, colors, out_dir, img_rgb = fs.names_of_food(
            img_path, masks.copy(), xyxy.copy(), names, confs
        )

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
            "img_rgb": img_rgb
        }

        return JSONResponse({
            "session_id": session_id,
            "overlay_image": str(overlay_path),
            "cropped_output": str(cropped_path),
            "classes_detected": classes,
            "confidences": confidences,
            "cropped_path_url":  cropped_image_url,
            "out_dir": str(out_dir),
            "ui_modes": ["ui1", "ui2", "ui3", "ui4", "ui6"]
        })
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Uploading Error: {str(e)}")

# --- Generate UI asset ---


class GeneratePayload(BaseModel):
    ui_mode: str
    session_id: str
    # image: str  # base64 PNG


@app.post("/create")
# async def create_ui(ui_mode: str, session_id: str = Query(...)):
async def create_ui(payload: GeneratePayload):
    session_id = payload.session_id
    ui_mode = payload.ui_mode

    print("SESSION ID IS", session_id)
    print("UI MODE IS", ui_mode)
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    media_path = None

    try:
        print("mode =", ui_mode)
        if ui_mode == "ui1":
            media_path = fs.createUI1(
                overlay_path=session["overlay_path"],
                classes=session["classes"],
                confs=session["confs"],
                colors=session["colors"],
                out_dir=Path(session["out_dir"])
            )

        elif ui_mode == "ui2":
            media_path = fs.createUI2(
                overlay_path=session["new_pth"],
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )

        elif ui_mode == "ui3":
            media_path = fs.createUI3(
                img_rgb=session["img_rgb"].copy(),
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )

        elif ui_mode == "ui4":
            img_rgb_cur = cv2.imread(fs.new_pth)
            print(type(img_rgb_cur))
            print("HEY, IMAGE IS PRESENT! and its dimensions are", img_rgb_cur.shape)
            media_path = fs.createUI5(
                img_rgb=img_rgb_cur.copy(),  # session["img_rgb"].copy(),
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )

        elif ui_mode == "ui6":
            print("this start")
            ori_img = cv2.imread(session["img_path"])
            img_rgb_cur = cv2.imread(fs.new_pth)
            print(type(img_rgb_cur))
            print("HEY, IMAGE IS PRESENT! and its dimensions are", img_rgb_cur.shape)
            media_path = fs.createUI6(
                ori_img=ori_img.copy(),
                img_rgb=img_rgb_cur.copy(),  # session["img_rgb"].copy(),
                classes=session["classes"],
                confs=session["confs"],
                class_masks=session["class_masks"],
                out_dir=Path(session["out_dir"])
            )
            print("this done")

        else:
            raise HTTPException(status_code=400, detail="Invalid UI mode")

        if not media_path:
            raise HTTPException(status_code=500, detail="UI generation failed")

        # Determine media type
        media_type = get_media_type(media_path)

        rel_path = Path(media_path).relative_to(OUTPUT_DIR)
        print("rel path is", rel_path)
        # pipeline_code_darpan_local_copy/v12/output_dir/ff1f11f2-bbfb-4edf-abf0-2deed9c9303c/ui6.mp4
        media_url = f"http://drishti.iiit.ac.in:8002/static/{rel_path}"
        # fs.undo_resizing(session["img_path"])
        return JSONResponse({
            "message": f"{ui_mode.upper()} created successfully.",
            "media_path": media_url,
            "media_type": media_type,
            "file_extension": Path(media_path).suffix.lower()
        })

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"UI creation error: {str(e)}")
