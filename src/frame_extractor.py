from pathlib import Path
import cv2

# FrameExtractor class handles the extraction of frames from a video file and 
# saves them as image files in a specified output directory.

class FrameExtractor:
    # Initializes the FrameExtractor with the video path and output directory.
    def __init__(self, video_path: str, output_dir: str | Path):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)

        if not self.video_path.exists() or not self.video_path.is_file():
            raise FileNotFoundError(f"Video file '{self.video_path}' does not exist or is not a file.")
        
        if not self.output_dir.exists() or not self.output_dir.is_dir():
             raise NotADirectoryError(f"Output directory '{self.output_dir}' does not exist or is not a directory.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Extracts frames from the video and saves them as image files in the output directory.
    def extract_frames(self, image_format: str = "png"):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Retrieves video properties for metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        saved_frames = 0
        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            filename = f"frame_{frame_index:05d}.{image_format}"
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), frame)
            saved_frames += 1
            frame_index += 1

        cap.release()

        # Return metadata about the frame extraction process
        return {
            "fps": fps,
            "video_total_frames": total_frames,
            "saved_frames": saved_frames,
            "width": width,
            "height": height,
            "image_format": image_format,
            "output_dir": str(self.output_dir),
        }