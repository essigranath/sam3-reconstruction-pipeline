from pathlib import Path
import shutil

# SubsetCreator class handles the creation of datasets by copying selected frames 
# and their corresponding SAM 3 masks to specified directories.

class SubsetCreator:
    # Initializes the SubsetCreator with the directory containing frames and the output directory for the datasets.
    def __init__(self, frames_dir: str | Path, original_dataset_dir: str | Path, masks_dir: str | Path):
        self.frames_dir = Path(frames_dir)
        self.original_dataset_dir = Path(original_dataset_dir)
        self.masks_dir = Path(masks_dir)

        if not self.frames_dir.exists() or not self.frames_dir.is_dir():
            raise FileNotFoundError(f"Frames directory '{self.frames_dir}' does not exist or is not a directory.")
        
        if not self.masks_dir.exists() or not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory '{self.masks_dir}' does not exist or is not a directory.")

        self.original_dataset_dir.mkdir(parents=True, exist_ok=True)

        if not self.original_dataset_dir.exists() or not self.original_dataset_dir.is_dir():
            raise NotADirectoryError(f"Original dataset directory '{self.original_dataset_dir}'does not exist or is not a directory.")
        
        self.sam3_dataset_dir = self.original_dataset_dir.parent / "sam3_dataset"

    # Creates datasets by copying selected subset frames and their corresponding SAM 3 masks 
    # to the specified directories, based on the given interval and image format.
    def create_datasets(self, interval: int = 10, image_format: str = 'png'):

        if interval <= 0:
            raise ValueError("Interval must be greater than 0.")
        
        if not image_format or not image_format.strip():
            raise ValueError("Image format cannot be empty.")

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory '{self.frames_dir}' does not exist.")
        
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory '{self.masks_dir}' does not exist.")

        sam3_frames_dir = self.sam3_dataset_dir / "frames"
        sam3_masks_dir = self.sam3_dataset_dir / "masks"

        sam3_frames_dir.mkdir(parents=True, exist_ok=True)
        sam3_masks_dir.mkdir(parents=True, exist_ok=True)

        if not sam3_frames_dir.exists() or not sam3_frames_dir.is_dir():
            raise NotADirectoryError(f"SAM3 frames directory '{sam3_frames_dir}' does not exist or is not a directory.")
        
        if not sam3_masks_dir.exists() or not sam3_masks_dir.is_dir():
            raise NotADirectoryError(f"SAM3 masks directory '{sam3_masks_dir}' does not exist or is not a directory.")

        frame_files = sorted(self.frames_dir.glob(f'*.{image_format}'))
        if not frame_files:
            raise RuntimeError(f"No '{image_format}' files found in '{self.frames_dir}'.")

        selected_frames = frame_files[::interval]

        if not selected_frames:
            raise RuntimeError(f"No frames selected with the given interval of {interval}.")

        copied_masks = 0

        for frame_path in selected_frames:

            original_output = self.original_dataset_dir / frame_path.name
            shutil.copy2(frame_path, original_output)

            if not original_output.exists():
                raise RuntimeError(f"Failed to copy frame '{frame_path}' to '{original_output}'.")

            sam3_frame_output = sam3_frames_dir / frame_path.name
            shutil.copy2(frame_path, sam3_frame_output)

            if not sam3_frame_output.exists():
                raise RuntimeError(f"Failed to copy frame '{frame_path}' to '{sam3_frame_output}'.")

            mask_name = frame_path.name
            mask_path = self.masks_dir / mask_name

            if mask_path.exists():
                sam3_mask_output = sam3_masks_dir / mask_name
                shutil.copy2(mask_path, sam3_mask_output)
                copied_masks += 1

            if copied_masks == 0:
                raise RuntimeError(f"No masks were copied. Please check if the masks exist in '{self.masks_dir}' and are named correctly.")

        metadata = {
            "interval": interval,
            "source_frame_dir": str(self.frames_dir),
            "source_masks_dir": str(self.masks_dir),
            "sam3_dataset_dir": str(self.sam3_dataset_dir),
            "original_dataset_dir": str(self.original_dataset_dir),
            "total_frames": len(frame_files),
            "selected_frames": len(selected_frames),
            "copied_masks": copied_masks,
            "image_format": image_format,
        }

        return metadata