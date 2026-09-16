from pathlib import Path
import json
from datetime import datetime

# ExperimentManager class manages the creation and organization of experiment runs,
# including directory structure, metadata, and logging.

class ExperimentManager:
    # Initializes the ExperimentManager with a specified root directory for experiments.
    def __init__(self, experiments_root="experiments"):
        self.experiments_root = Path(experiments_root)
        if self.experiments_root.exists() and not self.experiments_root.is_dir():
            raise NotADirectoryError(f"Experiments root is not a directory: {self.experiments_root}")
        self.experiments_root.mkdir(parents=True, exist_ok=True)

    # Creates a new experiment run with the specified video path and optional description, 
    # and sets up the necessary directory structure and metadata.
    def create_experiment_run(self, video_path: str, description: str = ""):
        video_path = Path(video_path)
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        experiment_id = self.__get_next_experiment_id()
        experiment_dir = self.experiments_root / experiment_id

        paths = {
            "experiment_dir": experiment_dir,
            "input": experiment_dir / "input",
            "frames": experiment_dir / "frames",
            "sam3_masks": experiment_dir / "sam3_masks",
            "datasets": experiment_dir / "datasets",
            "original_dataset": experiment_dir / "datasets" / "original_dataset",
            "logs": experiment_dir / "logs",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        if not experiment_dir.exists():
            raise RuntimeError(f"Experiment directory was not created: {experiment_dir}")    

        metadata = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(timespec='seconds'),
            "video_path": str(video_path),
            "source_video": video_path.name,
            "description": description,
            "status": "created",
            "frame_extraction": {},
            "sam3_processing": {},
            "dataset_status": {
                "original_created": True,
                "sam3_created": False,
                "photoshop_created": False
            },
        }

        metadata_path = experiment_dir / "metadata.json"
        log_path = paths["logs"] / "log.txt"

        self._write_metadata(metadata_path, metadata)
        if not metadata_path.exists():
            raise RuntimeError(f"Metadata file was not created: {metadata_path}")
        
        self._write_log(log_path, f"Experiment created: {experiment_id}")
        if not log_path.exists():
            raise RuntimeError(f"Log file was not created: {log_path}")

        return {
            "experiment_id": experiment_id,
            "paths": paths,
            "metadata_path": metadata_path,
            "log_path": log_path
        }

    # Updates the metadata file with new information provided in the updates dictionary.
    def update_metadata(self, metadata_path: Path, updates: dict):
        metadata_path = Path(metadata_path)
        if not metadata_path.exists() or not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in metadata file: {metadata_path}") from e
        metadata.update(updates)
        self._write_metadata(metadata_path, metadata)

    # Public method for writing log messages from other classes
    def log(self, log_path: Path, message: str):
        if not message:
            raise ValueError("Log message cannot be empty.")
        self._write_log(log_path, message)    

    # Private method to determine the next experiment ID based on existing directories 
    # and make sure it follows the naming pattern
    def __get_next_experiment_id(self):
        existing = [
            p.name for p in self.experiments_root.iterdir()
            if p.is_dir() and p.name.startswith("experiment_")
        ]
        if not existing:
            return "experiment_001"
        
        numbers = []
        for name in existing:
            try:
                numbers.append(int(name.split("_")[1]))
            except ValueError:
                pass

        next_number = max(numbers) + 1 if numbers else 1
        return f"experiment_{next_number:03d}"
    
    # Writes the metadata dictionary to a JSON file at the specified path, ensuring the directory exists.
    def _write_metadata(self, metadata_path: Path, metadata: dict):
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

    # Writes a log message to the specified log file, prepending a timestamp and ensuring the directory exists.
    def _write_log(self, log_path: Path, message: str):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec='seconds')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")