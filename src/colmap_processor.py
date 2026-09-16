from pathlib import Path
import subprocess
import time
import shutil

# COLMAPProcessor is a utility class for running COLMAP sparse reconstruction on a dataset of images,
# optionally using masks to ignore certain regions during feature extraction.
# It handles preparing mask files, executing COLMAP commands, and collecting statistics about the resulting sparse models.

class COLMAPProcessor:
    # Initializes the COLMAPProcessor with the path to the COLMAP executable.
    def __init__(self, colmap_path: str | Path = "C:/Tools/COLMAP/COLMAP.bat"):
        self.colmap_path = Path(colmap_path)

        if not self.colmap_path.exists():
            raise FileNotFoundError(f"COLMAP not found: {self.colmap_path}")

    # Prepares COLMAP-compatible mask files by copying them from the source mask directory to the output mask directory.
    def prepare_colmap_masks(self, image_dir: Path, mask_dir: Path, output_mask_dir: Path):
        # Creates COLMAP compatible mask filenames
        output_mask_dir.mkdir(parents=True, exist_ok=True)

        for image_file in sorted(image_dir.glob("*.png")):
            source_mask = mask_dir / image_file.name

            if not source_mask.exists():
                raise FileNotFoundError(f"Missing mask for image: {image_file.name}")

            target_mask = output_mask_dir / f"{image_file.name}.png"
            shutil.copy2(source_mask, target_mask)

    # Runs the full COLMAP sparse reconstruction pipeline for a given dataset of images, optionally using masks.
    def run_sparse_reconstruction(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        mask_dir: str | Path | None = None,
        database_name: str = "database.db",
        dataset_name: str = "unknown"
    ):
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        database_path = output_dir / database_name
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        feature_command = [
            str(self.colmap_path),
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
        ]

        colmap_mask_dir = None

        if mask_dir is not None:
            mask_dir = Path(mask_dir)

            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

            colmap_mask_dir = output_dir / "colmap_masks"

            self.prepare_colmap_masks(
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_mask_dir=colmap_mask_dir
            )

            feature_command.extend([
                "--ImageReader.mask_path", str(colmap_mask_dir)
            ])

        start_time = time.perf_counter()

        try:

            # Step 1: Extract image features
            self.run_command(feature_command)

            # Step 2: Match image features
            self.run_command([
                str(self.colmap_path),
                "exhaustive_matcher",
                "--database_path", str(database_path),
            ])

            # Step 3: Build sparse 3D reconstruction
            self.run_command([
                str(self.colmap_path),
                "mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
            ])

            selected_model, all_sparse_models = self.select_largest_sparse_model(sparse_dir)

            model_dir = selected_model["model_dir"]
            txt_model_dir = selected_model["txt_model_dir"]

            registered_images = selected_model["registered_images"]
            points3D = selected_model["points3D"]
            mean_error = selected_model["mean_reprojection_error"]

            export_path = output_dir / "model.ply"

            # Step 4: Export sparse model to PLY
            self.run_command([
                str(self.colmap_path),
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(export_path),
                "--output_type", "PLY",
            ])

            processing_time = time.perf_counter() - start_time

            # Step 5: Return a summary of the processing results
            return {
                "dataset_name": dataset_name,
                "status": "completed",
                "success": True,
                "uses_masks": mask_dir is not None,
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir) if mask_dir is not None else None,
                "colmap_mask_dir": str(colmap_mask_dir) if colmap_mask_dir is not None else None,
                "output_dir": str(output_dir),
                "database_path": str(database_path),
                "sparse_model_dir": str(model_dir),
                "selected_sparse_model": selected_model["model_name"],
                "created_sparse_models": len(all_sparse_models),

                "ply_model_path": str(export_path),
                "registered_images": registered_images,
                "points3D": points3D,
                "mean_reprojection_error": mean_error,

                "all_sparse_models": [
                    {
                        "model_name": m["model_name"],
                        "registered_images": m["registered_images"],
                        "points3D": m["points3D"],
                        "mean_reprojection_error": m["mean_reprojection_error"]
                    }
                    for m in all_sparse_models
                ],
                "processing_time_seconds": round(processing_time, 2),
                "error": None
            }

        # If any step fails, return a standardized error result with details about the failure.
        except Exception as e:
            return self._empty_result(
                dataset_name=dataset_name,
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                error="Failed to create any sparse model"
            )

    # Runs a COLMAP command and shows full error output if it fails
    def run_command(self, command):
        result = subprocess.run(
            command,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"COLMAP command failed:\n"
                f"{' '.join(command)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        return result

    # Returns a standardized result dict for failed processing attempts
    def _empty_result(self, dataset_name, image_dir, mask_dir, output_dir, error=None):
        # Returns a standardized result dict for failed processing attempts
        return {
            "dataset_name": dataset_name,
            "status": "failed",
            "success": False,
            "uses_masks": mask_dir is not None,
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir) if mask_dir else None,
            "database_path": str(Path(output_dir) / "database.db"),
            "sparse_model_dir": None,
            "ply_model_path": None,
            "registered_images": 0,
            "points3D": 0,
            "mean_reprojection_error": None,
            "processing_time_seconds": None,
            "error": error
        }

    # Reads the COLMAP sparse model statistics from the TXT files and returns the number of registered images,
    # number of 3D points, and mean reprojection error.
    def read_sparse_model_stats(self, sparse_model_dir):
        sparse_model_dir = Path(sparse_model_dir)

        images_file = sparse_model_dir / "images.txt"
        points_file = sparse_model_dir / "points3D.txt"

        if not images_file.exists() or not points_file.exists():
            return 0, 0, None
        
        registered_images = 0

        with open(images_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    registered_images += 1

        registered_images = registered_images // 2

        point_count = 0
        reprojection_errors = []

        with open(points_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) >= 8:
                    point_count += 1
                    reprojection_errors.append(float(parts[7]))
                    
        mean_error = (
            sum(reprojection_errors) / len(reprojection_errors)
            if reprojection_errors
            else None
        )

        return registered_images, point_count, mean_error

    # Selects the largest COLMAP sparse model based on registered images and 3D points.
    def select_largest_sparse_model(self, sparse_dir: Path):
        # Selects the largest COLMAP sparse model based on registered images
        model_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])

        if not model_dirs:
            raise RuntimeError("No sparse model was created by COLMAP")

        model_infos = []

        for model_dir in model_dirs:
            txt_model_dir = model_dir.parent.parent / f"sparse_txt_{model_dir.name}"
            txt_model_dir.mkdir(parents=True, exist_ok=True)

            self.run_command([
                str(self.colmap_path),
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(txt_model_dir),
                "--output_type", "TXT",
            ])

            registered_images, points3D, mean_error = self.read_sparse_model_stats(txt_model_dir)

            model_infos.append({
                "model_name": model_dir.name,
                "model_dir": model_dir,
                "txt_model_dir": txt_model_dir,
                "registered_images": registered_images,
                "points3D": points3D,
                "mean_reprojection_error": mean_error
            })

        selected_model = max(
            model_infos,
            key=lambda m: (m["registered_images"], m["points3D"])
        )

        return selected_model, model_infos