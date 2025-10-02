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

# Check for AI interpolation models
RIFE_AVAILABLE = False
FILM_AVAILABLE = False

try:
    # RIFE implementation check
    import importlib.util
    if importlib.util.find_spec("rife") is not None:
        RIFE_AVAILABLE = True
except:
    pass

try:
    # FILM implementation check  
    if importlib.util.find_spec("film") is not None:
        FILM_AVAILABLE = True
except:
    pass


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
                    "default": 0,  # 0 = unlimited
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "frame_step": ("INT", {
                    "default": 1,  # Load every Nth frame
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
    CATEGORY = "animation/loaders"
    
    def load_batch(self, source_type, path, file_filter, sort_method, 
                   max_frames, frame_step, random_seed=0):
        """
        Load image batch from various sources
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Parse file extensions
        extensions = [ext.strip().lower() for ext in file_filter.split(',')]
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Load images based on source type
        if source_type == "directory":
            image_paths = self.load_from_directory(path, extensions)
        elif source_type == "zip_file":
            image_paths = self.load_from_zip(path, extensions)
        elif source_type == "zip_url":
            image_paths = self.load_from_url(path, extensions)
        
        if not image_paths:
            raise ValueError(f"No images found matching extensions: {file_filter}")
        
        # Sort images
        if sort_method == "alphabetical":
            image_paths.sort()
        elif sort_method == "modified_date":
            image_paths.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
        elif sort_method == "random":
            random.seed(random_seed)
            random.shuffle(image_paths)
        
        # Apply frame step
        if frame_step > 1:
            image_paths = image_paths[::frame_step]
        
        # Apply max frames limit
        if max_frames > 0 and len(image_paths) > max_frames:
            image_paths = image_paths[:max_frames]
        
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
            raise ValueError("Failed to load any images")
        
        # Convert to torch tensor (B, H, W, C)
        images_tensor = torch.from_numpy(np.stack(images))
        
        return (images_tensor, len(images))
    
    def load_from_directory(self, directory, extensions):
        """Load images from a directory"""
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(directory).glob(f'*{ext}'))
        
        return [str(p) for p in image_paths]
    
    def load_from_zip(self, zip_path, extensions):
        """Load images from a ZIP file"""
        if not os.path.isfile(zip_path):
            raise ValueError(f"ZIP file not found: {zip_path}")
        
        temp_dir = tempfile.mkdtemp(prefix="comfyui_batch_")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only image files
                for file_info in zip_ref.filelist:
                    if any(file_info.filename.lower().endswith(ext) for ext in extensions):
                        zip_ref.extract(file_info, temp_dir)
            
            # Get all extracted image paths
            image_paths = []
            for ext in extensions:
                image_paths.extend(Path(temp_dir).rglob(f'*{ext}'))
            
            return [str(p) for p in image_paths]
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to extract ZIP: {e}")
    
    def load_from_url(self, url, extensions):
        """Download ZIP from URL and load images"""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available. Cannot download from URL.")
        
        print(f"Downloading ZIP from: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip.write(response.content)
            temp_zip.close()
            
            # Load from the downloaded ZIP
            image_paths = self.load_from_zip(temp_zip.name, extensions)
            
            # Clean up ZIP file
            os.unlink(temp_zip.name)
            
            return image_paths
        except requests.RequestException as e:
            raise ValueError(f"Failed to download ZIP from URL: {e}")


class BatchMultiFolderLoader:
    """
    Batch process multiple folders/ZIPs automatically
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parent_directory": ("STRING", {
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
            },
            "optional": {
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT",)
    RETURN_NAMES = ("all_batches", "batch_info", "total_batches",)
    FUNCTION = "load_multi_batch"
    CATEGORY = "animation/loaders"
    
    def load_multi_batch(self, parent_directory, source_type, file_filter, 
                        sort_method, max_frames_per_batch, frame_step, random_seed=0):
        """
        Load multiple batches from subfolders or ZIP files
        """
        if not parent_directory or not os.path.isdir(parent_directory):
            raise ValueError(f"Parent directory not found: {parent_directory}")
        
        parent_path = Path(parent_directory)
        sources = []
        
        # Collect sources based on type
        if source_type in ["subfolders", "both"]:
            # Get all subdirectories
            subdirs = [d for d in parent_path.iterdir() if d.is_dir()]
            sources.extend([(str(d), "folder") for d in subdirs])
        
        if source_type in ["zip_files", "both"]:
            # Get all ZIP files
            zip_files = list(parent_path.glob("*.zip"))
            sources.extend([(str(z), "zip") for z in zip_files])
        
        if not sources:
            raise ValueError(f"No sources found in {parent_directory}")
        
        # Sort sources alphabetically
        sources.sort(key=lambda x: x[0])
        
        print(f"Found {len(sources)} sources to process")
        
        # Parse extensions
        extensions = [ext.strip().lower() for ext in file_filter.split(',')]
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Process each source
        all_batches = []
        batch_info_list = []
        
        for source_path, source_kind in sources:
            source_name = os.path.basename(source_path)
            print(f"Processing: {source_name}")
            
            try:
                # Load images from source
                if source_kind == "folder":
                    image_paths = self.load_from_directory(source_path, extensions)
                else:  # zip
                    image_paths = self.load_from_zip(source_path, extensions)
                
                if not image_paths:
                    print(f"Warning: No images found in {source_name}")
                    continue
                
                # Sort images
                if sort_method == "alphabetical":
                    image_paths.sort()
                elif sort_method == "modified_date":
                    image_paths.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
                elif sort_method == "random":
                    random.seed(random_seed)
                    random.shuffle(image_paths)
                
                # Apply frame step
                if frame_step > 1:
                    image_paths = image_paths[::frame_step]
                
                # Apply max frames limit
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
                
                if images:
                    batch_tensor = torch.from_numpy(np.stack(images))
                    all_batches.append(batch_tensor)
                    batch_info_list.append({
                        "source": source_name,
                        "type": source_kind,
                        "frame_count": len(images)
                    })
                    print(f"Loaded {len(images)} frames from {source_name}")
            
            except Exception as e:
                print(f"Error processing {source_name}: {e}")
        
        if not all_batches:
            raise ValueError("No batches loaded successfully")
        
        # Concatenate all batches (with batch dimension preserved via metadata)
        # We'll concatenate but preserve batch boundaries in info
        combined_batches = torch.cat(all_batches, dim=0)
        batch_info_json = json.dumps(batch_info_list, indent=2)
        
        print(f"Total batches loaded: {len(all_batches)}")
        print(f"Total frames: {combined_batches.shape[0]}")
        
        return (combined_batches, batch_info_json, len(all_batches))
    
    def load_from_directory(self, directory, extensions):
        """Load images from a directory"""
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(directory).glob(f'*{ext}'))
        return [str(p) for p in image_paths]
    
    def load_from_zip(self, zip_path, extensions):
        """Load images from a ZIP file"""
        temp_dir = tempfile.mkdtemp(prefix="comfyui_multibatch_")
        
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


class VideoFrameExtractor:
    """
    Extract frames from video files with FPS control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "output_fps": ("FLOAT", {
                    "default": 0.0,  # 0 = keep original
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
                    "default": 0,  # 0 = all frames
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
    CATEGORY = "animation/loaders"
    
    def extract_frames(self, video_path, output_fps, start_frame, end_frame, frame_step):
        """
        Extract frames from video
        """
        if not video_path or not os.path.isfile(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if not CV2_AVAILABLE and not IMAGEIO_AVAILABLE:
            raise RuntimeError("Neither opencv-python nor imageio is available. Cannot extract video frames.")
        
        print(f"Extracting frames from: {video_path}")
        
        # Use OpenCV if available (faster), otherwise imageio
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
        
        # Convert to torch tensor
        frames_array = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_array)
        
        print(f"Extracted {len(frames)} frames (Original FPS: {original_fps})")
        
        return (frames_tensor, len(frames), original_fps)
    
    def extract_with_opencv(self, video_path, output_fps, start_frame, end_frame, frame_step):
        """Extract frames using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling
        if output_fps > 0 and output_fps != original_fps:
            # Resample to target FPS
            frame_interval = int(original_fps / output_fps)
            frame_interval = max(1, frame_interval)
        else:
            frame_interval = 1
        
        # Determine frame range
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
            
            # Check if frame is in range and matches sampling
            if (frame_idx >= start_frame and 
                frame_idx < end_frame and 
                (frame_idx - start_frame) % (frame_interval * frame_step) == 0):
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to 0-1
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
            
            frame_idx += 1
            
            if frame_idx >= end_frame:
                break
        
        cap.release()
        
        return frames, original_fps
    
    def extract_with_imageio(self, video_path, output_fps, start_frame, end_frame, frame_step):
        """Extract frames using imageio"""
        reader = imageio.get_reader(video_path)
        
        # Get metadata
        meta = reader.get_meta_data()
        original_fps = meta.get('fps', 30.0)
        
        # Calculate frame sampling
        if output_fps > 0 and output_fps != original_fps:
            frame_interval = int(original_fps / output_fps)
            frame_interval = max(1, frame_interval)
        else:
            frame_interval = 1
        
        frames = []
        
        try:
            for frame_idx, frame in enumerate(reader):
                # Check frame range
                if end_frame > 0 and frame_idx >= end_frame:
                    break
                
                if (frame_idx >= start_frame and 
                    (frame_idx - start_frame) % (frame_interval * frame_step) == 0):
                    
                    # Normalize to 0-1
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
        
        # Add AI methods if available
        if RIFE_AVAILABLE:
            interpolation_options.append("rife")
        if FILM_AVAILABLE:
            interpolation_options.append("film")
        
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
    CATEGORY = "animation"
    
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
            elif interpolation == "rife" and RIFE_AVAILABLE:
                final_frames = self.apply_rife(ordered_frames, interpolation_frames)
            elif interpolation == "film" and FILM_AVAILABLE:
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
                
                # Convert to grayscale for flow calculation
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Generate intermediate frames
                for j in range(1, num_blend_frames + 1):
                    alpha = j / (num_blend_frames + 1)
                    
                    # Warp frame1 towards frame2
                    h, w = frame1.shape[:2]
                    flow_scaled = flow * alpha
                    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (map_x + flow_scaled[..., 0]).astype(np.float32)
                    map_y = (map_y + flow_scaled[..., 1]).astype(np.float32)
                    
                    warped = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)
                    
                    # Blend warped frame with frame2
                    blended = cv2.addWeighted(warped, 1-alpha, frame2, alpha, 0)
                    
                    # Convert back to tensor
                    blended_tensor = torch.from_numpy(blended.astype(np.float32) / 255.0).unsqueeze(0)
                    result.append(blended_tensor)
        
        return torch.cat(result, dim=0)
    
    def apply_rife(self, frames, num_blend_frames):
        """RIFE-based interpolation (placeholder - requires RIFE model)"""
        print("RIFE interpolation: Model integration pending")
        # TODO: Implement RIFE model loading and inference
        # For now, fall back to optical flow or crossfade
        if CV2_AVAILABLE:
            return self.apply_optical_flow(frames, num_blend_frames)
        return self.apply_crossfade(frames, num_blend_frames)
    
    def apply_film(self, frames, num_blend_frames):
        """FILM-based interpolation (placeholder - requires FILM model)"""
        print("FILM interpolation: Model integration pending")
        # TODO: Implement FILM model loading and inference
        # For now, fall back to optical flow or crossfade
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
    CATEGORY = "animation"
    OUTPUT_NODE = True
    
    def encode_animation(self, frames, fps, loop_mode, 
                        loop_duration_seconds, loop_count, 
                        filename_prefix, output_directory, 
                        save_gif, save_mp4, save_webm, save_avi, save_mov, save_mkv,
                        quality=90, auto_open_folder="disabled"):
        """
        Encode frames to multiple animation formats
        """
        # Check if at least one format is enabled
        enabled_formats = [save_gif, save_mp4, save_webm, save_avi, save_mov, save_mkv]
        if all(fmt == "disabled" for fmt in enabled_formats):
            print("Warning: No output formats enabled")
            return {}
        
        # Calculate loop repetitions
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
        
        # Convert frames to numpy
        frames_np = (final_frames.cpu().numpy() * 255).astype(np.uint8)
        
        # Determine output directory
        if output_directory and output_directory.strip():
            output_dir = output_directory.strip()
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate base filename
        counter = 1
        while True:
            base_filename = f"{filename_prefix}_{counter:04d}"
            conflict = False
            
            # Check all enabled formats for conflicts
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
        
        # Save each enabled format
        saved_files = []
        
        if save_gif == "enabled":
            gif_path = os.path.join(output_dir, f"{base_filename}.gif")
            self.save_gif(frames_np, gif_path, fps, loop_gif, quality)
            saved_files.append(f"{base_filename}.gif")
            print(f"Saved GIF: {gif_path}")
        
        if save_mp4 == "enabled":
            if not IMAGEIO_AVAILABLE and not CV2_AVAILABLE:
                print("Warning: Cannot save MP4 - neither imageio nor opencv-python is available")
            else:
                mp4_path = os.path.join(output_dir, f"{base_filename}.mp4")
                self.save_video(frames_np, mp4_path, fps, quality, "mp4")
                saved_files.append(f"{base_filename}.mp4")
                print(f"Saved MP4: {mp4_path}")
        
        if save_webm == "enabled":
            if not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save WebM - imageio is not available")
            else:
                webm_path = os.path.join(output_dir, f"{base_filename}.webm")
                self.save_video(frames_np, webm_path, fps, quality, "webm")
                saved_files.append(f"{base_filename}.webm")
                print(f"Saved WebM: {webm_path}")
        
        if save_avi == "enabled":
            if not CV2_AVAILABLE and not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save AVI - neither opencv-python nor imageio is available")
            else:
                avi_path = os.path.join(output_dir, f"{base_filename}.avi")
                self.save_video(frames_np, avi_path, fps, quality, "avi")
                saved_files.append(f"{base_filename}.avi")
                print(f"Saved AVI: {avi_path}")
        
        if save_mov == "enabled":
            if not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save MOV - imageio is not available")
            else:
                mov_path = os.path.join(output_dir, f"{base_filename}.mov")
                self.save_video(frames_np, mov_path, fps, quality, "mov")
                saved_files.append(f"{base_filename}.mov")
                print(f"Saved MOV: {mov_path}")
        
        if save_mkv == "enabled":
            if not IMAGEIO_AVAILABLE:
                print("Warning: Cannot save MKV - imageio is not available")
            else:
                mkv_path = os.path.join(output_dir, f"{base_filename}.mkv")
                self.save_video(frames_np, mkv_path, fps, quality, "mkv")
                saved_files.append(f"{base_filename}.mkv")
                print(f"Saved MKV: {mkv_path}")
        
        # Auto-open folder if enabled
        if auto_open_folder == "enabled" and saved_files:
            self.open_folder(output_dir)
        
        # Print summary
        if saved_files:
            print(f"Animation encoding complete. Saved: {', '.join(saved_files)}")
        
        return {}
    
    def save_gif(self, frames, filepath, fps, loop, quality):
        """Save as GIF using Pillow"""
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
        """Save video in various formats using imageio or opencv"""
        if IMAGEIO_AVAILABLE:
            self.save_video_imageio(frames, filepath, fps, quality, format_type)
        elif CV2_AVAILABLE and format_type in ["mp4", "avi"]:
            self.save_video_opencv(frames, filepath, fps, format_type)
        else:
            print(f"Warning: Cannot save {format_type.upper()} - required library not available")
    
    def save_video_imageio(self, frames, filepath, fps, quality, format_type):
        """Save video using imageio"""
        codec_map = {
            "mp4": "libx264",
            "webm": "libvpx-vp9",
            "avi": "png",  # Lossless codec for AVI
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
        """Save video using opencv (fallback for MP4/AVI)"""
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
        """Open folder in system file explorer"""
        import platform
        import subprocess
        
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(folder_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            print(f"Could not open folder: {e}")


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdvancedBatchLoader": AdvancedBatchLoader,
    "BatchMultiFolderLoader": BatchMultiFolderLoader,
    "VideoFrameExtractor": VideoFrameExtractor,
    "BatchAnimationProcessor": BatchAnimationProcessor,
    "MultiFormatAnimationEncoder": MultiFormatAnimationEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedBatchLoader": "Advanced Batch Loader (Dir/ZIP/URL)",
    "BatchMultiFolderLoader": "Batch Multi-Folder/ZIP Loader",
    "VideoFrameExtractor": "Video Frame Extractor",
    "BatchAnimationProcessor": "Batch Animation Processor",
    "MultiFormatAnimationEncoder": "Multi-Format Animation Encoder",
}