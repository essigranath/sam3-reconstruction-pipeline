from pathlib import Path
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor

# SAM3Processor class handles the processing of a video using the SAM3 model, 
# including starting a session, adding prompts, propagating masks, and saving the resulting masks to an output directory.

class SAM3Processor:
    # Initialize the SAM3Processor with the video path and output directory
    def __init__(self, video_path: str | Path, output_dir: str | Path, log_callback=None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)

        if not self.video_path.exists() or not self.video_path.is_file():
            raise FileNotFoundError(f"Video file '{self.video_path}' does not exist.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.output_dir.exists():
            raise NotADirectoryError(f"Output directory '{self.output_dir}' is not a valid directory.")
        
        self.log_callback = log_callback

        torch.cuda.empty_cache()
        self.video_predictor = build_sam3_video_predictor()
        
    # Process the video using SAM3, saving masks for each frame and returning metadata about the process
    def process_video(self, point: tuple[int, int] | None = None,
    text_prompt: str | None = None, start_frame_index: int = 0):
        if point is None and not text_prompt:
            raise ValueError("Either point or text_prompt must be provided.")
        if point is not None and text_prompt:
            raise ValueError("Use either point or text_prompt, not both.")
        if start_frame_index < 0:
            raise ValueError("Start frame index cannot be negative.")
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file '{self.video_path}' does not exist.")
        
        self.log("Starting SAM3 session...")

        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(self.video_path),
            )
        )

        if "session_id" not in response:
            raise RuntimeError("Failed to start SAM3 session: 'session_id' not found in response.")

        session_id = response["session_id"]
        self.log("SAM3 session started.")

        # Add the prompt (point or text) to the SAM3 session
        if point is not None:
            x, y = point

            points = np.array([[x, y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            self.log(f"Adding point prompt at x={x}, y={y}...")
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=start_frame_index,
                    obj_id=1,
                    points=points,
                    point_labels=point_labels,
                )
            )

            self.log("Point prompt added.")

        else:
            self.log(f"Adding text prompt: {text_prompt}...")
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=start_frame_index,
                    obj_id=1,
                    text=text_prompt,
                )
            )

            self.log("Text prompt added.")
        self.log("Starting SAM3 propagation...")

        saved_masks = 0
        processed_frames = 0
        frames_with_multiple_masks = 0
        extra_masks_ignored = 0

        # Process the video frames and save the resulting masks to the output directory
        for frame_output in self.video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=start_frame_index,
                propagation_direction="forward",
            )
        ):
            processed_frames += 1

            # Check if the expected keys are present in the frame output
            if "frame_index" not in frame_output or "outputs" not in frame_output:
                self.log("Warning: Received frame output without 'frame_index' or 'outputs'. Skipping this output.")
                continue

            frame_index = frame_output["frame_index"]
            output = frame_output["outputs"]

            self.log(f"Processing SAM3 frame {frame_index}...")

            # Check if the expected key 'out_binary_masks' is present in the output
            if "out_binary_masks" not in output:
                self.log(f"Warning: SAM3 output for frame {frame_index} is missing out_binary_masks. Skipping this frame.")
                continue

            masks = output["out_binary_masks"]

            # Check if masks is a list and contains at least one mask
            if len(masks) > 1:
                frames_with_multiple_masks += 1
                extra_masks_ignored += len(masks) - 1
                self.log(
                    f"Warning: SAM3 returned {len(masks)} masks for frame {frame_index}. "
                    "Only the first mask is saved."
                )

            mask = masks[0]

            # Check if the mask is empty
            mask_array = np.squeeze(mask)
            if mask_array.size == 0:
                raise RuntimeError(f"Received an empty mask for frame {frame_index}.")

            mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
            save_path = self.output_dir / f"frame_{frame_index:05d}.png"

            mask_image.save(save_path)

            if not save_path.exists():
                raise RuntimeError(f"Failed to save mask image: {save_path}")

            saved_masks += 1
            self.log(f"\nSaved mask: {save_path}")

        self.log(f"SAM3 propagation completed. Processed {processed_frames} frames.")

        if saved_masks == 0:
            raise RuntimeError("No masks were saved during SAM3 processing. Please check the input video and prompt.")  

        self.log(f"SAM3 processing completed. Saved {saved_masks} masks.")      

        # Return metadata about the SAM3 processing
        metadata = {
            "prompt_type": "point" if point is not None else "text",
            "point": point,
            "text_prompt": text_prompt,
            "start_frame_index": start_frame_index,
            "video_path": str(self.video_path),
            "output_dir": str(self.output_dir),
            "saved_masks": saved_masks,
            "frames_with_multiple_masks": frames_with_multiple_masks,
            "extra_masks_ignored": extra_masks_ignored,
            "status": "completed",
        }

        return metadata
    
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)