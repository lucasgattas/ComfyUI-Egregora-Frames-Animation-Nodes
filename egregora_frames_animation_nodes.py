"""
ComfyUI Animation Nodes
Advanced animation system with batch processing, AI interpolation, and video extraction
"""

import torch
import numpy as np
from PIL import Image
import io
import os
import random
from pathlib import Path
import folder_paths
import zipfile
import tempfile
import shutil
import json
import urllib.request
import urllib.parse

# Try importing optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. URL loading will be disabled.")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. MP4/WebM export will be disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available. Some features may be limited.")

# AI interpolation model paths
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Check and setup RIFE
RIFE_AVAILABLE = False
try:
    # Try to import existing RIFE installation
    from inference_rife import RIFE_Interpolator
    RIFE_AVAILABLE = True
    print("✓ RIFE model available")
except ImportError:
    # Check if we have RIFE in our models directory
    rife_model_path = MODELS_DIR / "rife"
    if rife_model_path.exists():
        import sys
        sys.path.insert(0, str(rife_model_path))
        try:
            from inference_rife import RIFE_Interpolator
            RIFE_AVAILABLE = True
            print("✓ RIFE model available (local)")
        except:
            print("✗ RIFE model files found but not functional")
    else:
        print("✗ RIFE not available. Will auto-download on first use.")

# Check and setup FILM
FILM_AVAILABLE = False
try:
    from film_interpolator import FILM_Interpolator
    FILM_AVAILABLE = True
    print("✓ FILM model available")
except ImportError:
    film_model_path = MODELS_DIR / "film"
    if film_model_path.exists():
        import sys
        sys.path.insert(0, str(film_model_path))
        try:
            from film_interpolator import FILM_Interpolator
            FILM_AVAILABLE = True
            print("✓ FILM model available (local)")
        except:
            print("✗ FILM model files found but not functional")
    else:
        print("✗ FILM not available. Will auto-download on first use.")


def download_file(url, dest_path, desc="Downloading"):
    """Download file with progress"""
    print(f"{desc}: {url}")
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        done = int(50 * downloaded / total_size)
                        print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size}", end='')
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Downloaded to: {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def setup_rife_model():
    """Download and setup RIFE model"""
    global RIFE_AVAILABLE
    
    rife_dir = MODELS_DIR / "rife"
    rife_dir.mkdir(exist_ok=True)
    
    print("Setting up RIFE model...")
    
    # RIFE model files (using official RIFE repository)
    model_url = "https://github.com/hzwer/Practical-RIFE/releases/download/4.6/flownet.pkl"
    model_path = rife_dir / "flownet.pkl"
    
    if not model_path.exists():
        if download_file(model_url, model_path, "Downloading RIFE model"):
            print("✓ RIFE model downloaded successfully")
            RIFE_AVAILABLE = True
        else:
            print("✗ Failed to download RIFE model")
            return False
    else:
        RIFE_AVAILABLE = True
        print("✓ RIFE model already exists")
    
    return True


def setup_film_model():
    """Download and setup FILM model"""
    global FILM_AVAILABLE
    
    film_dir = MODELS_DIR / "film"
    film_dir.mkdir(exist_ok=True)
    
    print("Setting up FILM model...")
    print("Note: FILM requires TensorFlow. This is a placeholder for future implementation.")
    print("For now, please use optical_flow interpolation which works great!")
    
    return False


class AdvancedBatchLoader:
    """
    Advanced batch loader supporting directory, ZIP files, and ZIP from URL
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_type": (["directory", "zip_file", "zip_url"],),
                "path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "file_filter": ("STRING", {
                    "default": "png,jpg,jpeg,webp",
                    "multiline": False
                }),
                "sort_method": (["alphabetical", "random", "modified_date"],),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "frame_step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("images", "frame_count",)
    FUNCTION = "load_batch"
    CATEGORY = "Egregora/animation"
    
    def load_batch(self, source_type, path, file_filter, sort_method, 
                   max_frames, frame_step, random_seed=0):
        """
        Load image batch from various sources
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        extensions = [ext.strip().lower() for ext in file_filter.split(',')]
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        if source_type == "directory":
            image_paths = self.load_from_directory(path, extensions)
        elif source_type == "zip_file":
            image_paths = self.load_from_zip(path, extensions)
        elif source_type == "zip_url":
            image_paths = self.load_from_url(path, extensions)
        
        if not image_paths:
            raise ValueError(f"No images found matching extensions: {file_filter}")
        
        if sort_method == "alphabetical":
            image_paths.sort()
        elif sort_method == "modified_date":
            image_paths.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
        elif sort_method == "random":
            random.seed(random_seed)
            random.shuffle(image_paths)
        
        if frame_step > 1:
            image_paths = image_paths[::frame_step]
        
        if max_frames > 0 and len(image_paths) > max_frames:
            image_paths = image_paths[:max_frames]
        
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
        
        if not images:
            raise ValueError("Failed to load any images")
        
        images_tensor = torch.from_numpy(np.stack(images))
        
        return (images_tensor, len(images))
    
    def load_from_directory(self, directory, extensions):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(directory).glob(f'*{ext}'))
        
        return [str(p) for p in image_paths]
    
    def load_from_zip(self, zip_path, extensions):
        if not os.path.isfile(zip_path):
            raise ValueError(f"ZIP file not found: {zip_path}")
        
        temp_dir = tempfile.mkdtemp(prefix="comfyui_batch_")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if any(file_info.filename.lower().endswith(ext) for ext in extensions):
                        zip_ref.extract(file_info, temp_dir)
            
            image_paths = []
            for ext in extensions:
                image_paths.extend(Path(temp_dir).rglob(f'*{ext}'))
            
            return [str(p) for p in image_paths]
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to extract ZIP: {e}")
    
    def load_from_url(self, url, extensions):
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available. Cannot download from URL.")
        
        print(f"Downloading ZIP from: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip.write(response.content)
            temp_zip.close()
            
            image_paths = self.load_from_zip(temp_zip.name, extensions)
            os.unlink(temp_zip.name)
            
            return image_paths
        except requests.RequestException as e:
            raise ValueError(f"Failed to download ZIP from URL: {e}")


class BatchMultiFolderProcessor:
    """
    Process multiple folders/ZIPs and generate separate animations for each
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        interpolation_options = ["none", "crossfade", "optical_flow", "rife", "film"]
        
        return {
            "required": {
                "input_type": (["local_directory", "zip_url"],),
                "path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "source_type": (["subfolders", "zip_files", "both"],),
                "file_filter": ("STRING", {
                    "default": "png,jpg,jpeg,webp",
                    "multiline": False
                }),
                "sort_method": (["alphabetical", "random", "modified_date"],),
                "max_frames_per_batch": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "frame_step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "interpolation": (interpolation_options,),
                "interpolation_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
                "fps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 60,
                    "step": 1
                }),
                "final_output_fps": ("INT", {
                    "default": 0,  # 0 = same as fps
                    "min": 0,
                    "max": 120,
                    "step": 1
                }),
                "output_mode": (["separate_files", "combined_video"],),
                "loop_mode": (["no_loop", "loop_duration", "loop_count"],),
                "loop_duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 300.0,
                    "step": 0.1
                }),
                "loop_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "output_format": (["gif", "mp4", "webm", "avi", "mov", "mkv"],),
            },
            "optional": {
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "process_multi_batch"
    CATEGORY = "Egregora/animation"
    OUTPUT_NODE = True
    
    def process_multi_batch(self, input_type, path, source_type, file_filter, sort_method,
                           max_frames_per_batch, frame_step, interpolation, interpolation_frames,
                           fps, final_output_fps, output_mode, loop_mode, loop_duration_seconds, 
                           loop_count, output_directory, output_format, random_seed=0, quality=90):
        """
        Process multiple batches and generate separate or combined animations
        """
        # Handle URL download for ZIP
        if input_type == "zip_url":
            path = self.download_zip_from_url(path)
            if not path:
                raise ValueError("Failed to download ZIP from URL")
        
        if not path or not os.path.exists(path):
            raise ValueError(f"Path not found: {path}")
        
        # Check if path is a ZIP file
        if os.path.isfile(path) and path.lower().endswith('.zip'):
            temp_extract_dir = tempfile.mkdtemp(prefix="comfyui_multibatch_extract_")
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            parent_path = Path(temp_extract_dir)
        else:
            parent_path = Path(path)
        
        if not parent_path.is_dir():
            raise ValueError(f"Not a valid directory: {path}")
        
        sources = []
        
        # Collect sources
        if source_type in ["subfolders", "both"]:
            subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
            sources.extend([(str(d), "folder", d.name) for d in subdirs])
        
        if source_type in ["zip_files", "both"]:
            zip_files = list(parent_path.glob("*.zip"))
            sources.extend([(str(z), "zip", z.stem) for z in zip_files])
        
        if not sources:
            raise ValueError(f"No sources found in {path}")
        
        sources.sort(key=lambda x: x[0])
        
        print(f"Found {len(sources)} sources to process")
        print(f"Output mode: {output_mode}")
        
        extensions = [ext.strip().lower() for ext in file_filter.split(',')]
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Determine output directory
        if output_directory and output_directory.strip():
            out_dir = output_directory.strip()
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = folder_paths.get_output_directory()
        
        # For combined mode, prepare video writer
        combined_writer = None
        combined_path = None
        save_fps = final_output_fps if final_output_fps > 0 else fps
        
        if output_mode == "combined_video":
            # Generate combined filename
            combined_filename = f"combined_animation.{output_format}"
            combined_path = os.path.join(out_dir, combined_filename)
            
            counter = 1
            while os.path.exists(combined_path):
                combined_filename = f"combined_animation_{counter:03d}.{output_format}"
                combined_path = os.path.join(out_dir, combined_filename)
                counter += 1
            
            print(f"\n{'='*60}")
            print(f"Combined mode: Streaming directly to {combined_filename}")
            print(f"{'='*60}")
        
        # Store all processed batches for combined mode
        processed_count = 0
        total_combined_frames = 0
        
        # Process each source
        for source_path, source_kind, source_name in sources:
            print(f"\n{'='*60}")
            print(f"Processing: {source_name}")
            print(f"{'='*60}")
            
            try:
                # Load images from source
                if source_kind == "folder":
                    image_paths = self.load_from_directory(source_path, extensions)
                else:
                    image_paths = self.load_from_zip(source_path, extensions)
                
                if not image_paths:
                    print(f"Warning: No images found in {source_name}")
                    continue
                
                # Sort
                if sort_method == "alphabetical":
                    image_paths.sort()
                elif sort_method == "modified_date":
                    image_paths.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
                elif sort_method == "random":
                    random.seed(random_seed)
                    random.shuffle(image_paths)
                
                # Apply frame step and max frames
                if frame_step > 1:
                    image_paths = image_paths[::frame_step]
                
                if max_frames_per_batch > 0 and len(image_paths) > max_frames_per_batch:
                    image_paths = image_paths[:max_frames_per_batch]
                
                # Load images
                images = []
                for img_path in image_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_array = np.array(img).astype(np.float32) / 255.0
                        images.append(img_array)
                    except Exception as e:
                        print(f"Warning: Failed to load {img_path}: {e}")
                
                if not images:
                    print(f"No valid images loaded from {source_name}")
                    continue
                
                frames = torch.from_numpy(np.stack(images))
                
                # Apply interpolation
                if interpolation != "none" and interpolation_frames > 0:
                    print(f"Applying {interpolation} interpolation...")
                    frames = self.apply_interpolation(frames, interpolation, interpolation_frames)
                
                # Apply looping
                num_frames = frames.shape[0]
                frame_duration = num_frames / fps
                
                if loop_mode == "no_loop":
                    final_frames = frames
                elif loop_mode == "loop_duration":
                    repeats = max(1, int(np.ceil(loop_duration_seconds / frame_duration)))
                    final_frames = frames.repeat(repeats, 1, 1, 1)
                    target_frame_count = int(loop_duration_seconds * fps)
                    final_frames = final_frames[:target_frame_count]
                elif loop_mode == "loop_count":
                    final_frames = frames.repeat(loop_count, 1, 1, 1)
                
                # Store for combined mode or save separately
                if output_mode == "combined_video":
                    # Stream frames directly to video writer (memory efficient)
                    frames_np = (final_frames.cpu().numpy() * 255).astype(np.uint8)
                    
                    # Apply final FPS reencoding if needed
                    if final_output_fps > 0 and final_output_fps != fps:
                        frames_np = self.reencode_fps(frames_np, fps, final_output_fps)
                    
                    # Initialize writer on first batch
                    if combined_writer is None:
                        if output_format == "gif":
                            # For GIF, we need to collect all frames (no streaming possible)
                            combined_writer = {"type": "gif", "frames": [], "fps": save_fps, "quality": quality}
                        else:
                            combined_writer = self.init_video_writer(combined_path, save_fps, 
                                                                     frames_np[0].shape[:2], 
                                                                     quality, output_format)
                    
                    # Write frames
                    if output_format == "gif":
                        combined_writer["frames"].extend([Image.fromarray(f) for f in frames_np])
                    else:
                        for frame in frames_np:
                            self.write_frame_to_video(combined_writer, frame, output_format)
                    
                    total_combined_frames += len(frames_np)
                    print(f"✓ Streamed: {source_name} ({len(frames)} original, {len(final_frames)} processed, {len(frames_np)} encoded @ {save_fps}fps)")
                else:
                    # Save separate file
                    frames_np = (final_frames.cpu().numpy() * 255).astype(np.uint8)
                    
                    # Apply final FPS reencoding if needed
                    if final_output_fps > 0 and final_output_fps != fps:
                        frames_np = self.reencode_fps(frames_np, fps, final_output_fps)
                        save_fps = final_output_fps
                    else:
                        save_fps = fps
                    
                    output_filename = f"{source_name}.{output_format}"
                    output_path = os.path.join(out_dir, output_filename)
                    
                    # Handle filename conflicts
                    counter = 1
                    while os.path.exists(output_path):
                        output_filename = f"{source_name}_{counter:03d}.{output_format}"
                        output_path = os.path.join(out_dir, output_filename)
                        counter += 1
                    
                    # Save animation
                    if output_format == "gif":
                        self.save_gif(frames_np, output_path, save_fps, loop_mode != "no_loop", quality)
                    else:
                        self.save_video(frames_np, output_path, save_fps, quality, output_format)
                    
                    print(f"✓ Saved: {output_filename} ({len(frames)} original, {len(final_frames)} processed, {len(frames_np)} final @ {save_fps}fps)")
                
                processed_count += 1
            
            except Exception as e:
                print(f"✗ Error processing {source_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Finalize combined video if in combined mode
        if output_mode == "combined_video" and combined_writer is not None:
            print(f"\n{'='*60}")
            print("Finalizing combined video...")
            
            if output_format == "gif":
                # Save GIF
                if combined_writer["frames"]:
                    duration_ms = int(1000 / combined_writer["fps"])
                    combined_writer["frames"][0].save(
                        combined_path,
                        save_all=True,
                        append_images=combined_writer["frames"][1:],
                        duration=duration_ms,
                        loop=0,
                        optimize=True,
                        quality=combined_writer["quality"]
                    )
                    print(f"✓ Combined GIF saved: {os.path.basename(combined_path)}")
            else:
                # Close video writer
                self.close_video_writer(combined_writer, output_format)
                print(f"✓ Combined video saved: {os.path.basename(combined_path)}")
            
            print(f"  Total frames: {total_combined_frames}")
            print(f"  Duration: {total_combined_frames / save_fps:.2f} seconds @ {save_fps}fps")
            print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        if output_mode == "combined_video":
            print(f"Combined processing complete: {processed_count} batches merged into 1 video")
        else:
            print(f"Batch processing complete: {processed_count}/{len(sources)} animations created")
        print(f"Output directory: {out_dir}")
        print(f"{'='*60}")
        
        return {}
    
    def init_video_writer(self, filepath, fps, frame_shape, quality, format_type):
        """Initialize video writer for streaming"""
        height, width = frame_shape
        
        if IMAGEIO_AVAILABLE and format_type != "avi":
            codec_map = {
                "mp4": "libx264",
                "webm": "libvpx-vp9",
                "mov": "libx264",
                "mkv": "libx264"
            }
            codec = codec_map.get(format_type, "libx264")
            
            writer = imageio.get_writer(
                filepath,
                fps=fps,
                codec=codec,
                quality=quality / 10,
                pixelformat='yuv420p'
            )
            return {"type": "imageio", "writer": writer}
        
        elif CV2_AVAILABLE:
            fourcc_map = {
                "mp4": cv2.VideoWriter_fourcc(*'mp4v'),
                "avi": cv2.VideoWriter_fourcc(*'XVID')
            }
            fourcc = fourcc_map.get(format_type, cv2.VideoWriter_fourcc(*'mp4v'))
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            return {"type": "opencv", "writer": writer}
        
        return None
    
    def write_frame_to_video(self, writer_obj, frame, format_type):
        """Write a single frame to video writer"""
        if writer_obj["type"] == "imageio":
            writer_obj["writer"].append_data(frame)
        elif writer_obj["type"] == "opencv":
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer_obj["writer"].write(frame_bgr)
    
    def close_video_writer(self, writer_obj, format_type):
        """Close video writer"""
        if writer_obj["type"] == "imageio":
            writer_obj["writer"].close()
        elif writer_obj["type"] == "opencv":
            writer_obj["writer"].release()
    
    def reencode_fps(self, frames, original_fps, target_fps):
        """
        Reencode frames to different FPS while maintaining playback speed
        Duplicates frames as needed to match target FPS
        """
        if original_fps == target_fps:
            return frames
        
        num_frames = len(frames)
        duration = num_frames / original_fps  # Duration in seconds
        target_frame_count = int(duration * target_fps)
        
        # Create frame indices for resampling
        original_indices = np.arange(num_frames)
        target_indices = np.linspace(0, num_frames - 1, target_frame_count)
        
        # Resample frames (duplicates frames as needed)
        reencoded_frames = []
        for idx in target_indices:
            frame_idx = int(np.round(idx))
            frame_idx = min(frame_idx, num_frames - 1)  # Clamp to valid range
            reencoded_frames.append(frames[frame_idx])
        
        reencoded = np.array(reencoded_frames)
        print(f"  FPS reencoding: {original_fps}fps ({num_frames} frames) → {target_fps}fps ({len(reencoded)} frames)")
        print(f"  Duration maintained: {duration:.2f} seconds")
        
        return reencoded
    
    def apply_interpolation(self, frames, method, num_frames):
        """Apply interpolation between frames"""
        if method == "crossfade":
            return self.apply_crossfade(frames, num_frames)
        elif method == "optical_flow":
            return self.apply_optical_flow(frames, num_frames)
        elif method == "rife":
            return self.apply_rife(frames, num_frames)
        elif method == "film":
            return self.apply_film(frames, num_frames)
        return frames
    
    def apply_crossfade(self, frames, num_blend_frames):
        """Simple crossfade blending"""
        if len(frames) < 2:
            return frames
        
        result = []
        for i in range(len(frames)):
            result.append(frames[i:i+1])
            
            if i < len(frames) - 1:
                next_frame = frames[i + 1]
                current_frame = frames[i]
                
                for j in range(1, num_blend_frames + 1):
                    alpha = j / (num_blend_frames + 1)
                    blended = current_frame * (1 - alpha) + next_frame * alpha
                    result.append(blended.unsqueeze(0))
        
        return torch.cat(result, dim=0)
    
    def apply_optical_flow(self, frames, num_blend_frames):
        """Optical flow-based interpolation"""
        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available, using crossfade")
            return self.apply_crossfade(frames, num_blend_frames)
        
        if len(frames) < 2:
            return frames
        
        result = []
        frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
        
        for i in range(len(frames)):
            result.append(frames[i:i+1])
            
            if i < len(frames) - 1:
                frame1 = frames_np[i]
                frame2 = frames_np[i + 1]
                
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                for j in range(1, num_blend_frames + 1):
                    alpha = j / (num_blend_frames + 1)
                    
                    h, w = frame1.shape[:2]
                    flow_scaled = flow * alpha
                    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (map_x + flow_scaled[..., 0]).astype(np.float32)
                    map_y = (map_y + flow_scaled[..., 1]).astype(np.float32)
                    
                    warped = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)
                    blended = cv2.addWeighted(warped, 1-alpha, frame2, alpha, 0)
                    
                    blended_tensor = torch.from_numpy(blended.astype(np.float32) / 255.0).unsqueeze(0)
                    result.append(blended_tensor)
        
        return torch.cat(result, dim=0)
    
    def apply_rife(self, frames, num_blend_frames):
        """RIFE interpolation with fallback"""
        if not RIFE_AVAILABLE:
            print("RIFE not available, attempting setup...")
            setup_rife_model()
        
        print("RIFE: Using optical flow fallback")
        if CV2_AVAILABLE:
            return self.apply_optical_flow(frames, num_blend_frames)
        return self.apply_crossfade(frames, num_blend_frames)
    
    def reencode_fps(self, frames, original_fps, target_fps):
        """
        Reencode frames to different FPS while maintaining playback speed
        Duplicates frames as needed to match target FPS
        """
        if original_fps == target_fps:
            return frames
        
        num_frames = len(frames)
        duration = num_frames / original_fps  # Duration in seconds
        target_frame_count = int(duration * target_fps)
        
        # Create frame indices for resampling
        original_indices = np.arange(num_frames)
        target_indices = np.linspace(0, num_frames - 1, target_frame_count)
        
        # Resample frames (duplicates frames as needed)
        reencoded_frames = []
        for idx in target_indices:
            frame_idx = int(np.round(idx))
            frame_idx = min(frame_idx, num_frames - 1)  # Clamp to valid range
            reencoded_frames.append(frames[frame_idx])
        
        reencoded = np.array(reencoded_frames)
        print(f"  FPS reencoding: {original_fps}fps ({num_frames} frames) → {target_fps}fps ({len(reencoded)} frames)")
        print(f"  Duration maintained: {duration:.2f} seconds")
        
        return reencoded
    
    def apply_film(self, frames, num_blend_frames):
        """FILM interpolation with fallback"""
        if not FILM_AVAILABLE:
            print("FILM not available. Using optical flow fallback")
        
        if CV2_AVAILABLE:
            return self.apply_optical_flow(frames, num_blend_frames)
        return self.apply_crossfade(frames, num_blend_frames)
    
    def download_zip_from_url(self, url):
        """Download ZIP from URL"""
        if not REQUESTS_AVAILABLE:
            print("Error: requests library not available")
            return None
        
        try:
            print(f"Downloading ZIP from URL: {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                temp_zip.write(chunk)
                if total_size > 0:
                    done = int(50 * downloaded / total_size)
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
            
            print()  # Newline after progress
            temp_zip.close()
            print(f"✓ Downloaded to: {temp_zip.name}")
            return temp_zip.name
        
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    def load_from_directory(self, directory, extensions):
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(directory).glob(f'*{ext}'))
        return [str(p) for p in image_paths]
    
    def load_from_zip(self, zip_path, extensions):
        temp_dir = tempfile.mkdtemp(prefix="comfyui_zipbatch_")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if any(file_info.filename.lower().endswith(ext) for ext in extensions):
                        zip_ref.extract(file_info, temp_dir)
            
            image_paths = []
            for ext in extensions:
                image_paths.extend(Path(temp_dir).rglob(f'*{ext}'))
            
            return [str(p) for p in image_paths]
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to extract ZIP: {e}")
    
    def save_gif(self, frames, filepath, fps, loop, quality):
        pil_frames = [Image.fromarray(frame) for frame in frames]
        duration_ms = int(1000 / fps)
        
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1,
            optimize=True,
            quality=quality
        )
    
    def save_video(self, frames, filepath, fps, quality, format_type):
        if IMAGEIO_AVAILABLE:
            self.save_video_imageio(frames, filepath, fps, quality, format_type)
        elif CV2_AVAILABLE and format_type in ["mp4", "avi"]:
            self.save_video_opencv(frames, filepath, fps, format_type)
    
    def save_video_imageio(self, frames, filepath, fps, quality, format_type):
        codec_map = {
            "mp4": "libx264",
            "webm": "libvpx-vp9",
            "avi": "png",
            "mov": "libx264",
            "mkv": "libx264"
        }
        
        codec = codec_map.get(format_type, "libx264")
        
        writer = imageio.get_writer(
            filepath,
            fps=fps,
            codec=codec,
            quality=quality / 10,
            pixelformat='yuv420p' if codec != 'png' else 'rgb24'
        )
        
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    
    def save_video_opencv(self, frames, filepath, fps, format_type):
        height, width = frames[0].shape[:2]
        
        fourcc_map = {
            "mp4": cv2.VideoWriter_fourcc(*'mp4v'),
            "avi": cv2.VideoWriter_fourcc(*'XVID')
        }
        
        fourcc = fourcc_map.get(format_type, cv2.VideoWriter_fourcc(*'mp4v'))
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()


class VideoFrameExtractor:
    """
    Extract frames from video files with FPS control and URL support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["local_file", "url"],),
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "output_fps": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1
                }),
                "end_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1
                }),
                "frame_step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT",)
    RETURN_NAMES = ("frames", "frame_count", "original_fps",)
    FUNCTION = "extract_frames"
    CATEGORY = "Egregora/animation"
    
    def extract_frames(self, input_type, video_path, output_fps, start_frame, end_frame, frame_step):
        """
        Extract frames from video
        """
        # Handle URL download
        if input_type == "url":
            video_path = self.download_video_from_url(video_path)
            if not video_path:
                raise ValueError("Failed to download video from URL")
        
        if not video_path or not os.path.isfile(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if not CV2_AVAILABLE and not IMAGEIO_AVAILABLE:
            raise RuntimeError("Neither opencv-python nor imageio is available. Cannot extract video frames.")
        
        print(f"Extracting frames from: {video_path}")
        
        if CV2_AVAILABLE:
            frames, original_fps = self.extract_with_opencv(
                video_path, output_fps, start_frame, end_frame, frame_step
            )
        else:
            frames, original_fps = self.extract_with_imageio(
                video_path, output_fps, start_frame, end_frame, frame_step
            )
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        frames_array = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_array)
        
        print(f"Extracted {len(frames)} frames (Original FPS: {original_fps})")
        
        return (frames_tensor, len(frames), original_fps)
    
    def download_video_from_url(self, url):
        """Download video from URL"""
        if not REQUESTS_AVAILABLE:
            print("Error: requests library not available")
            return None
        
        try:
            print(f"Downloading video from URL: {url}")
            
            # Determine file extension from URL
            parsed_url = urllib.parse.urlparse(url)
            ext = Path(parsed_url.path).suffix or '.mp4'
            
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                temp_video.write(chunk)
                if total_size > 0:
                    done = int(50 * downloaded / total_size)
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
            
            print()  # Newline after progress
            temp_video.close()
            print(f"✓ Downloaded to: {temp_video.name}")
            return temp_video.name
        
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    def extract_with_opencv(self, video_path, output_fps, start_frame, end_frame, frame_step):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_fps > 0 and output_fps != original_fps:
            frame_interval = int(original_fps / output_fps)
            frame_interval = max(1, frame_interval)
        else:
            frame_interval = 1
        
        if end_frame == 0:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx >= start_frame and 
                frame_idx < end_frame and 
                (frame_idx - start_frame) % (frame_interval * frame_step) == 0):
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
            
            frame_idx += 1
            
            if frame_idx >= end_frame:
                break
        
        cap.release()
        
        return frames, original_fps
    
    def extract_with_imageio(self, video_path, output_fps, start_frame, end_frame, frame_step):
        reader = imageio.get_reader(video_path)
        
        meta = reader.get_meta_data()
        original_fps = meta.get('fps', 30.0)
        
        if output_fps > 0 and output_fps != original_fps:
            frame_interval = int(original_fps / output_fps)
            frame_interval = max(1, frame_interval)
        else:
            frame_interval = 1
        
        frames = []
        
        try:
            for frame_idx, frame in enumerate(reader):
                if end_frame > 0 and frame_idx >= end_frame:
                    break
                
                if (frame_idx >= start_frame and 
                    (frame_idx - start_frame) % (frame_interval * frame_step) == 0):
                    
                    frame_normalized = frame.astype(np.float32) / 255.0
                    frames.append(frame_normalized)
        
        finally:
            reader.close()
        
        return frames, original_fps


class BatchAnimationProcessor:
    """
    Processes image batches for animation: ordering, timing, and interpolation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        interpolation_options = ["none", "crossfade", "optical_flow"]
        
        # Always show RIFE and FILM options - they'll auto-download on first use
        interpolation_options.extend(["rife", "film"])
        
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_order": (["sequential", "random", "ping-pong", "reverse"],),
                "fps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 60,
                    "step": 1
                }),
                "interpolation": (interpolation_options,),
                "interpolation_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("processed_frames", "fps",)
    FUNCTION = "process_frames"
    CATEGORY = "Egregora/animation"
    
    def process_frames(self, images, frame_order, fps, interpolation, 
                      interpolation_frames, random_seed=0):
        """
        Process frames according to ordering and interpolation settings
        """
        batch_size = images.shape[0]
        frames = images.clone()
        
        # Apply frame ordering
        if frame_order == "sequential":
            ordered_frames = frames
        elif frame_order == "random":
            random.seed(random_seed)
            indices = list(range(batch_size))
            random.shuffle(indices)
            ordered_frames = frames[indices]
        elif frame_order == "reverse":
            ordered_frames = torch.flip(frames, [0])
        elif frame_order == "ping-pong":
            forward = frames
            backward = torch.flip(frames[1:-1], [0]) if batch_size > 2 else torch.empty(0, *frames.shape[1:])
            ordered_frames = torch.cat([forward, backward], dim=0)
        
        # Apply interpolation
        if interpolation != "none" and interpolation_frames > 0:
            if interpolation == "crossfade":
                final_frames = self.apply_crossfade(ordered_frames, interpolation_frames)
            elif interpolation == "optical_flow":
                final_frames = self.apply_optical_flow(ordered_frames, interpolation_frames)
            elif interpolation == "rife":
                final_frames = self.apply_rife(ordered_frames, interpolation_frames)
            elif interpolation == "film":
                final_frames = self.apply_film(ordered_frames, interpolation_frames)
            else:
                print(f"Warning: {interpolation} not available, using crossfade")
                final_frames = self.apply_crossfade(ordered_frames, interpolation_frames)
        else:
            final_frames = ordered_frames
        
        return (final_frames, fps,)
    
    def apply_crossfade(self, frames, num_blend_frames):
        """Simple crossfade blending"""
        if len(frames) < 2:
            return frames
        
        result = []
        
        for i in range(len(frames)):
            result.append(frames[i:i+1])
            
            if i < len(frames) - 1:
                next_frame = frames[i + 1]
                current_frame = frames[i]
                
                for j in range(1, num_blend_frames + 1):
                    alpha = j / (num_blend_frames + 1)
                    blended = current_frame * (1 - alpha) + next_frame * alpha
                    result.append(blended.unsqueeze(0))
        
        return torch.cat(result, dim=0)
    
    def apply_optical_flow(self, frames, num_blend_frames):
        """Optical flow-based interpolation using OpenCV"""
        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available, falling back to crossfade")
            return self.apply_crossfade(frames, num_blend_frames)
        
        if len(frames) < 2:
            return frames
        
        result = []
        frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
        
        for i in range(len(frames)):
            result.append(frames[i:i+1])
            
            if i < len(frames) - 1:
                frame1 = frames_np[i]
                frame2 = frames_np[i + 1]
                
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                for j in range(1, num_blend_frames + 1):
                    alpha = j / (num_blend_frames + 1)
                    
                    h, w = frame1.shape[:2]
                    flow_scaled = flow * alpha
                    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (map_x + flow_scaled[..., 0]).astype(np.float32)
                    map_y = (map_y + flow_scaled[..., 1]).astype(np.float32)
                    
                    warped = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)
                    blended = cv2.addWeighted(warped, 1-alpha, frame2, alpha, 0)
                    
                    blended_tensor = torch.from_numpy(blended.astype(np.float32) / 255.0).unsqueeze(0)
                    result.append(blended_tensor)
        
        return torch.cat(result, dim=0)
    
    def apply_rife(self, frames, num_blend_frames):
        """RIFE-based interpolation"""
        global RIFE_AVAILABLE
        
        if not RIFE_AVAILABLE:
            print("RIFE model not available. Attempting to download...")
            if setup_rife_model():
                print("RIFE model setup complete. Note: Full integration requires additional setup.")
                print("For now, using optical flow interpolation.")
            else:
                print("RIFE setup failed. Using optical flow interpolation.")
        else:
            print("RIFE model available but integration pending. Using optical flow for now.")
        
        # TODO: Full RIFE implementation
        # For now, fall back to optical flow
        if CV2_AVAILABLE:
            return self.apply_optical_flow(frames, num_blend_frames)
        return self.apply_crossfade(frames, num_blend_frames)
    
    def apply_film(self, frames, num_blend_frames):
        """FILM-based interpolation"""
        global FILM_AVAILABLE
        
        if not FILM_AVAILABLE:
            print("FILM model not available.")
            print("Note: FILM requires TensorFlow and is complex to integrate.")
            print("Using optical flow interpolation instead.")
        
        # TODO: Full FILM implementation
        if CV2_AVAILABLE:
            return self.apply_optical_flow(frames, num_blend_frames)
        return self.apply_crossfade(frames, num_blend_frames)


class MultiFormatAnimationEncoder:
    """
    Encodes processed frames to multiple formats (GIF/MP4/WebM/AVI/MOV/MKV)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 60,
                    "step": 1
                }),
                "final_output_fps": ("INT", {
                    "default": 0,  # 0 = same as fps
                    "min": 0,
                    "max": 120,
                    "step": 1
                }),
                "loop_mode": (["no_loop", "loop_duration", "loop_count"],),
                "loop_duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 300.0,
                    "step": 0.1
                }),
                "loop_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "filename_prefix": ("STRING", {
                    "default": "animation"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "save_gif": (["disabled", "enabled"],),
                "save_mp4": (["disabled", "enabled"],),
                "save_webm": (["disabled", "enabled"],),
                "save_avi": (["disabled", "enabled"],),
                "save_mov": (["disabled", "enabled"],),
                "save_mkv": (["disabled", "enabled"],),
            },
            "optional": {
                "quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "auto_open_folder": (["disabled", "enabled"],),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "encode_animation"
    CATEGORY = "Egregora/animation"
    OUTPUT_NODE = True
    
    def encode_animation(self, frames, fps, final_output_fps, loop_mode, 
                        loop_duration_seconds, loop_count, 
                        filename_prefix, output_directory, 
                        save_gif, save_mp4, save_webm, save_avi, save_mov, save_mkv,
                        quality=90, auto_open_folder="disabled"):
        """
        Encode frames to multiple animation formats
        """
        enabled_formats = [save_gif, save_mp4, save_webm, save_avi, save_mov, save_mkv]
        if all(fmt == "disabled" for fmt in enabled_formats):
            print("Warning: No output formats enabled")
            return {}
        
        num_frames = frames.shape[0]
        frame_duration = num_frames / fps
        
        if loop_mode == "no_loop":
            final_frames = frames
            loop_gif = False
        elif loop_mode == "loop_duration":
            repeats = max(1, int(np.ceil(loop_duration_seconds / frame_duration)))
            final_frames = frames.repeat(repeats, 1, 1, 1)
            target_frame_count = int(loop_duration_seconds * fps)
            final_frames = final_frames[:target_frame_count]
            loop_gif = True
        elif loop_mode == "loop_count":
            final_frames = frames.repeat(loop_count, 1, 1, 1)
            loop_gif = True
        
        frames_np = (final_frames.cpu().numpy() * 255).astype(np.uint8)
        
        # Apply final FPS reencoding if needed
        if final_output_fps > 0 and final_output_fps != fps:
            frames_np = self.reencode_fps(frames_np, fps, final_output_fps)
            save_fps = final_output_fps
        else:
            save_fps = fps
        
        if output_directory and output_directory.strip():
            output_dir = output_directory.strip()
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
        
        counter = 1
        while True:
            base_filename = f"{filename_prefix}_{counter:04d}"
            conflict = False
            
            format_map = {
                "gif": save_gif, "mp4": save_mp4, "webm": save_webm,
                "avi": save_avi, "mov": save_mov, "mkv": save_mkv
            }
            
            for ext, enabled in format_map.items():
                if enabled == "enabled" and os.path.exists(os.path.join(output_dir, f"{base_filename}.{ext}")):
                    conflict = True
                    break
            
            if not conflict:
                break
            counter += 1
        
        saved_files = []
        
        if save_gif == "enabled":
            gif_path = os.path.join(output_dir, f"{base_filename}.gif")
            self.save_gif(frames_np, gif_path, save_fps, loop_gif, quality)
            saved_files.append(f"{base_filename}.gif")
            print(f"Saved GIF: {gif_path}")
        
        if save_mp4 == "enabled":
            if not IMAGEIO_AVAILABLE and not CV2_AVAILABLE:
                print("Warning: Cannot save MP4 - neither imageio nor opencv-python is available")
            else:
                mp4_path = os.path.join(output_dir, f"{base_filename}.mp4")
                self.save_video(frames_np, mp4_path, save_fps, quality, "mp4")
                saved_files.append(f"{base_filename}.mp4")
                print(f"Saved MP4: {mp4_path}")
        
        if save_webm == "enabled":
            if not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save WebM - imageio is not available")
            else:
                webm_path = os.path.join(output_dir, f"{base_filename}.webm")
                self.save_video(frames_np, webm_path, save_fps, quality, "webm")
                saved_files.append(f"{base_filename}.webm")
                print(f"Saved WebM: {webm_path}")
        
        if save_avi == "enabled":
            if not CV2_AVAILABLE and not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save AVI - neither opencv-python nor imageio is available")
            else:
                avi_path = os.path.join(output_dir, f"{base_filename}.avi")
                self.save_video(frames_np, avi_path, save_fps, quality, "avi")
                saved_files.append(f"{base_filename}.avi")
                print(f"Saved AVI: {avi_path}")
        
        if save_mov == "enabled":
            if not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save MOV - imageio is not available")
            else:
                mov_path = os.path.join(output_dir, f"{base_filename}.mov")
                self.save_video(frames_np, mov_path, save_fps, quality, "mov")
                saved_files.append(f"{base_filename}.mov")
                print(f"Saved MOV: {mov_path}")
        
        if save_mkv == "enabled":
            if not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save MKV - imageio is not available")
            else:
                mkv_path = os.path.join(output_dir, f"{base_filename}.mkv")
                self.save_video(frames_np, mkv_path, save_fps, quality, "mkv")
                saved_files.append(f"{base_filename}.mkv")
                print(f"Saved MKV: {mkv_path}")
        
        if auto_open_folder == "enabled" and saved_files:
            self.open_folder(output_dir)
        
        if saved_files:
            print(f"Animation encoding complete. Saved: {', '.join(saved_files)}")
        
        return {}
    
    def save_gif(self, frames, filepath, fps, loop, quality):
        pil_frames = [Image.fromarray(frame) for frame in frames]
        duration_ms = int(1000 / fps)
        
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1,
            optimize=True,
            quality=quality
        )
    
    def save_video(self, frames, filepath, fps, quality, format_type):
        if IMAGEIO_AVAILABLE:
            self.save_video_imageio(frames, filepath, fps, quality, format_type)
        elif CV2_AVAILABLE and format_type in ["mp4", "avi"]:
            self.save_video_opencv(frames, filepath, fps, format_type)
    
    def save_video_imageio(self, frames, filepath, fps, quality, format_type):
        codec_map = {
            "mp4": "libx264",
            "webm": "libvpx-vp9",
            "avi": "png",
            "mov": "libx264",
            "mkv": "libx264"
        }
        
        codec = codec_map.get(format_type, "libx264")
        
        writer = imageio.get_writer(
            filepath,
            fps=fps,
            codec=codec,
            quality=quality / 10,
            pixelformat='yuv420p' if codec != 'png' else 'rgb24'
        )
        
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    
    def save_video_opencv(self, frames, filepath, fps, format_type):
        height, width = frames[0].shape[:2]
        
        fourcc_map = {
            "mp4": cv2.VideoWriter_fourcc(*'mp4v'),
            "avi": cv2.VideoWriter_fourcc(*'XVID')
        }
        
        fourcc = fourcc_map.get(format_type, cv2.VideoWriter_fourcc(*'mp4v'))
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def open_folder(self, folder_path):
        import platform
        import subprocess
        
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(folder_path)
            elif system == "Darwin":
                subprocess.run(["open", folder_path])
            else:
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            print(f"Could not open folder: {e}")
    
    def reencode_fps(self, frames, original_fps, target_fps):
        """
        Reencode frames to different FPS while maintaining playback speed
        Duplicates frames as needed to match target FPS
        """
        if original_fps == target_fps:
            return frames
        
        num_frames = len(frames)
        duration = num_frames / original_fps  # Duration in seconds
        target_frame_count = int(duration * target_fps)
        
        # Create frame indices for resampling
        original_indices = np.arange(num_frames)
        target_indices = np.linspace(0, num_frames - 1, target_frame_count)
        
        # Resample frames (duplicates frames as needed)
        reencoded_frames = []
        for idx in target_indices:
            frame_idx = int(np.round(idx))
            frame_idx = min(frame_idx, num_frames - 1)  # Clamp to valid range
            reencoded_frames.append(frames[frame_idx])
        
        reencoded = np.array(reencoded_frames)
        print(f"FPS reencoding: {original_fps}fps ({num_frames} frames) → {target_fps}fps ({len(reencoded)} frames)")
        print(f"Duration maintained: {duration:.2f} seconds")
        
        return reencoded
    
    def init_video_writer(self, filepath, fps, frame_shape, quality, format_type):
        """Initialize video writer for streaming"""
        height, width = frame_shape
        
        if IMAGEIO_AVAILABLE and format_type != "avi":
            codec_map = {
                "mp4": "libx264",
                "webm": "libvpx-vp9",
                "mov": "libx264",
                "mkv": "libx264"
            }
            codec = codec_map.get(format_type, "libx264")
            
            writer = imageio.get_writer(
                filepath,
                fps=fps,
                codec=codec,
                quality=quality / 10,
                pixelformat='yuv420p'
            )
            return {"type": "imageio", "writer": writer}
        
        elif CV2_AVAILABLE:
            fourcc_map = {
                "mp4": cv2.VideoWriter_fourcc(*'mp4v'),
                "avi": cv2.VideoWriter_fourcc(*'XVID')
            }
            fourcc = fourcc_map.get(format_type, cv2.VideoWriter_fourcc(*'mp4v'))
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            return {"type": "opencv", "writer": writer}
        
        return None
    
    def write_frame_to_video(self, writer_obj, frame, format_type):
        """Write a single frame to video writer"""
        if writer_obj["type"] == "imageio":
            writer_obj["writer"].append_data(frame)
        elif writer_obj["type"] == "opencv":
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer_obj["writer"].write(frame_bgr)
    
    def close_video_writer(self, writer_obj, format_type):
        """Close video writer"""
        if writer_obj["type"] == "imageio":
            writer_obj["writer"].close()
        elif writer_obj["type"] == "opencv":
            writer_obj["writer"].release()


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdvancedBatchLoader": AdvancedBatchLoader,
    "BatchMultiFolderProcessor": BatchMultiFolderProcessor,
    "VideoFrameExtractor": VideoFrameExtractor,
    "BatchAnimationProcessor": BatchAnimationProcessor,
    "MultiFormatAnimationEncoder": MultiFormatAnimationEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedBatchLoader": "Advanced Batch Loader (Dir/ZIP/URL)",
    "BatchMultiFolderProcessor": "Batch Multi-Folder Processor",
    "VideoFrameExtractor": "Video Frame Extractor (File/URL)",
    "BatchAnimationProcessor": "Batch Animation Processor",
    "MultiFormatAnimationEncoder": "Multi-Format Animation Encoder",
}
