import shutil
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
import sys
import threading
from PIL import Image, ImageTk
import cv2
import time
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiment_manager import ExperimentManager
from src.frame_extractor import FrameExtractor
from src.sam3_processor import SAM3Processor
from src.subset_creator import SubsetCreator
from src.colmap_processor import COLMAPProcessor

class ThesisApp:
    # This is the main class for the GUI application.
    # It initializes the GUI, handles user interactions, and manages the experiment workflow.
    def __init__(self, root):
        self.root = root
        self.root.title("SAM3 Pipeline")
        self.root.geometry("1100x750")
        self.root.minsize(950, 650)
        self.root.configure(bg="#f4f4f4")

        # Initialize all variables for storing user selections and experiment state
        self.video_path = None
        self.prompt_type = None
        self.text_prompt = None
        self.selected_point = None
        self.original_width = None
        self.original_height = None
        self.display_width = 500
        self.display_height = 300
        self.tk_image = None
        self.point_selection_start_time = None
        self.point_selection_end_time = None
        self.sam3_start_time = None
        self.sam3_end_time = None
        self.active_prompt_duration = None
        self.sam3_processing_duration = None
        self.text_prompt_start_time = None
        self.text_prompt_end_time = None
        self.text_prompt_confirmed = False
        self.photoshop_start_time = None
        self.photoshop_end_time = None
        self.photoshop_duration = None
        self.photoshop_masks_dir = None
        self.photoshop_frames_dir = None
        self.colmap_results_dir = None
        self.colmap_start_time = None
        self.colmap_end_time = None
        self.colmap_duration = None
        self.current_metadata_path = None
        self.current_log_path = None

        self.create_widgets()

    # This method creates all the GUI widgets and layouts them in the main window.
    def create_widgets(self):

        # Main container
        main = tk.Frame(self.root, bg="#f4f4f4")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Title
        header = tk.Frame(main, bg="#f4f4f4")
        header.pack(fill="x", pady=(0,10))

        title = tk.Label(
            header,
            text="SAM3 Pipeline",
            font=("Arial", 16, "bold"),
            bg="#f4f4f4"
        )
        title.pack(side="left")

        # Content area with left and right panels.
        content = tk.Frame(main, bg="#f4f4f4")
        content.pack(fill="both", expand=True)

        # Scrollable left panel for user inputs.
        left_container = tk.Frame(content, bg="#f4f4f4")
        left_container.pack(side="left", fill="both", expand=True, padx=(0, 10))

        left_canvas = tk.Canvas(
            left_container,
            bg="#f4f4f4",
            highlightthickness=0
        )
        left_canvas.pack(side="left", fill="both", expand=True)

        left_scrollbar = tk.Scrollbar(
            left_container,
            orient="vertical",
            command=left_canvas.yview
        )
        left_scrollbar.pack(side="right", fill="y")

        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_panel = tk.Frame(left_canvas, bg="#f4f4f4")

        left_window = left_canvas.create_window(
            (0, 0),
            window=left_panel,
            anchor="nw"
        )

        def update_scroll_region(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def resize_left_panel(event):
            left_canvas.itemconfig(left_window, width=event.width)

        left_panel.bind("<Configure>", update_scroll_region)
        left_canvas.bind("<Configure>", resize_left_panel)

        def on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_left_mousewheel(event):
            left_canvas.bind_all("<MouseWheel>", on_left_mousewheel)

        def unbind_left_mousewheel(event):
            left_canvas.unbind_all("<MouseWheel>")

        left_container.bind("<Enter>", bind_left_mousewheel)
        left_container.bind("<Leave>", unbind_left_mousewheel)

        # Scrollable right panel for the log.
        right_panel = tk.Frame(content, bg="#f4f4f4")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Select video section.
        video_frame = tk.LabelFrame(
            left_panel,
            text="1. Video Selection",
            padx=10,
            pady=10,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        video_frame.pack(fill="x", pady=5)

        self.video_label = tk.Label(
            video_frame,
            text="No video selected",
            anchor="w",
            bg="white",
            relief="sunken",
            width=45
        )
        self.video_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        select_button = tk.Button(
            video_frame,
            text="Select video",
            command=self.select_video,
            width=12
        )
        select_button.pack(side="right")

        # Capture conditions section for lighting, background, object description, and object structure.
        conditions_frame = tk.LabelFrame(
            left_panel,
            text="2. Capture Conditions",
            padx=10,
            pady=10,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        conditions_frame.pack(fill="x", pady=5)
        
        tk.Label(conditions_frame, text="Lighting:", bg="#f4f4f4").grid(row=0, column=0, sticky="w", pady=4)

        self.lighting_entry = ttk.Combobox(
            conditions_frame,
            values=[
                "Bright lighting",
                "Dim lighting",
            ],
            state="readonly",
            width=37
        )
        self.lighting_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.lighting_entry.set("Bright lighting")


        tk.Label(conditions_frame, text="Background:", bg="#f4f4f4").grid(row=1, column=0, sticky="w", pady=4)

        self.background_entry = ttk.Combobox(
            conditions_frame,
            values=[
                "Uncluttered",
                "Cluttered"
            ],
            state="readonly",
            width=37
        )
        self.background_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self.background_entry.set("Uncluttered")

        tk.Label(conditions_frame, text="Object description:", bg="#f4f4f4").grid(row=2, column=0, sticky="w", pady=4)
        self.object_entry = tk.Entry(conditions_frame, width=40)
        self.object_entry.grid(row=2, column=1, sticky="ew", pady=4)

        tk.Label(conditions_frame, text="Object structure:", bg="#f4f4f4").grid(row=3, column=0, sticky="w", pady=4)

        self.object_structure_entry = ttk.Combobox(
            conditions_frame,
            values=[
                "Simple",
                "Moderate",
                "Complex"
            ],
            state="readonly",
            width=37
        )
        self.object_structure_entry.grid(row=3, column=1, sticky="ew", pady=4)
        self.object_structure_entry.set("Simple")

        conditions_frame.columnconfigure(1, weight=1)

        # Prompt selection section for point or text prompt.
        point_frame = tk.LabelFrame(
            left_panel,
            text="3. SAM3 Prompt",
            padx=10,
            pady=10,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        point_frame.pack(fill="both", expand=True, pady=5)

        prompt_type_frame = tk.Frame(point_frame, bg="#f4f4f4")
        prompt_type_frame.pack(fill="x", pady=(0, 10))

        self.prompt_type_entry = ttk.Combobox(
            prompt_type_frame,
            values=["Point prompt", "Text prompt"],
            state="readonly",
            width=20
        )
        self.prompt_type_entry.pack(side="left")
        self.prompt_type_entry.set("Point prompt")

        text_prompt_section = tk.Frame(point_frame, bg="#f4f4f4")
        text_prompt_section.pack(fill="x", padx=(15, 0), pady=(0, 10))

        # Text prompt entry and buttons for starting and confirming the text prompt.
        tk.Label(
            text_prompt_section,
            text="Text prompt",
            bg="#f4f4f4",
            font=("Arial", 9, "bold")
        ).pack(anchor="w", pady=(0, 5))

        text_prompt_frame = tk.Frame(text_prompt_section, bg="#f4f4f4")
        text_prompt_frame.pack(fill="x", pady=(0, 8))

        self.text_prompt_entry = tk.Entry(text_prompt_frame, width=40)
        self.text_prompt_entry.pack(side="left")

        text_prompt_buttons = tk.Frame(text_prompt_section, bg="#f4f4f4")
        text_prompt_buttons.pack(fill="x")

        self.start_text_prompt_button = tk.Button(
            text_prompt_buttons,
            text="Start text prompt",
            command=self.start_text_prompt_timer,
            width=18
        )
        self.start_text_prompt_button.pack(side="left", padx=(0, 5))

        self.confirm_text_prompt_button = tk.Button(
            text_prompt_buttons,
            text="Confirm text prompt",
            command=self.confirm_text_prompt,
            width=20
        )
        self.confirm_text_prompt_button.pack(side="left")

        point_left = tk.Frame(point_frame, bg="#f4f4f4")
        point_left.pack(side="left", fill="y", padx=(0, 10))

        point_right = tk.Frame(point_frame, bg="#f4f4f4")
        point_right.pack(side="right", fill="both", expand=True)

        # Point prompt entry and buttons for starting and confirming the point prompt.
        tk.Label(
            point_right,
            text="Point selection",
            bg="#f4f4f4",
            font=("Arial", 9, "bold")
        ).pack(anchor="w", pady=(10, 5))

        self.frame_canvas = tk.Canvas(
            point_right,
            width=self.display_width,
            height=self.display_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.frame_canvas.pack(fill="both", expand=True)
        self.frame_canvas.bind("<Button-1>", self.on_frame_click)

        point_bottom = tk.Frame(point_right, bg="#f4f4f4")
        point_bottom.pack(fill="x", pady=(10, 5))

        self.point_label = tk.Label(
            point_bottom,
            text="Selected point: None",
            bg="#f4f4f4"
        )
        self.point_label.pack(side="left", padx=(0, 15))

        start_point_button = tk.Button(
            point_bottom,
            text="Start point selection",
            command=self.start_point_selection,
            width=18
        )
        start_point_button.pack(side="left", padx=(0, 5))

        confirm_point_button = tk.Button(
            point_bottom,
            text="Confirm Point",
            command=self.confirm_point_selection,
            width=15
        )
        confirm_point_button.pack(side="left")

        # Subset settings section for specifying the interval of frames to select for the dataset.
        subset_frame = tk.LabelFrame(
            left_panel,
            text="4. Dataset Settings",
            padx=10,
            pady=10,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        subset_frame.pack(fill="x", pady=5)

        tk.Label(subset_frame, text="Subset interval (frames):", bg="#f4f4f4").pack(side="left")

        self.interval_entry = tk.Entry(subset_frame, width=10)
        self.interval_entry.pack(side="left", padx=10)
        self.interval_entry.insert(0, "10")

        # Run experiment section with a button to start the SAM3 pipeline.
        run_frame = tk.LabelFrame(
            left_panel,
            text="5. Create experiment and run SAM3",
            padx=10,
            pady=15,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        run_frame.pack(fill="x", pady=5)

        run_button = tk.Button(
            run_frame,
            text="Run SAM3 pipeline",
            command=self.start_experiment_thread,
            width=25,
            height=2
        )
        run_button.pack()

        # Photoshop phase section for timing mask creation and validating the masks.
        photoshop_frame = tk.LabelFrame(
            left_panel,
            text="6. Photoshop Phase",
            padx=10,
            pady=10,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        photoshop_frame.pack(fill="x", pady=5)

        self.photoshop_status_label = tk.Label(
            photoshop_frame,
            text="Status: Not ready",
            bg="#f4f4f4",
            anchor="w"
        )
        self.photoshop_status_label.pack(fill="x", pady=(0, 5))

        tk.Label(
            photoshop_frame,
            text="Create Photoshop masks for the subset frames.",
            bg="#f4f4f4",
            anchor="w"
        ).pack(fill="x", pady=(0, 5))

        self.photoshop_folder_label = tk.Label(
            photoshop_frame,
            text="Masks folder: -",
            bg="white",
            relief="sunken",
            anchor="w",
            wraplength=500,
            justify="left"
        )
        self.photoshop_folder_label.pack(fill="x", pady=(0, 8))

        photoshop_buttons = tk.Frame(photoshop_frame, bg="#f4f4f4")
        photoshop_buttons.pack(fill="x")

        self.start_photoshop_button = tk.Button(
            photoshop_buttons,
            text="Start timer",
            command=self.start_photoshop_timer,
            width=15,
            state="disabled"
        )
        self.start_photoshop_button.pack(side="left", padx=(0, 5))

        self.stop_photoshop_button = tk.Button(
            photoshop_buttons,
            text="Stop timer",
            command=self.stop_photoshop_timer,
            width=15,
            state="disabled"
        )
        self.stop_photoshop_button.pack(side="left", padx=(0, 5))

        self.open_photoshop_folder_button = tk.Button(
            photoshop_buttons,
            text="Open folder",
            command=self.open_photoshop_folder,
            width=15,
            state="disabled"
        )
        self.open_photoshop_folder_button.pack(side="left", padx=(0, 5))

        self.validate_photoshop_button = tk.Button(
            photoshop_buttons,
            text="Validate masks",
            command=self.validate_photoshop_masks,
            width=15,
            state="disabled"
        )
        self.validate_photoshop_button.pack(side="left")

        # COLMAP phase section for running 3D reconstruction for all datasets using COLMAP.
        colmap_frame = tk.LabelFrame(
            left_panel,
            text="7. 3D Reconstruction",
            padx=10,
            pady=10,
            bg="#f4f4f4",
            font=("Arial", 10, "bold")
        )
        colmap_frame.pack(fill="x", pady=5)

        self.colmap_status_label = tk.Label(
            colmap_frame,
            text="Status: Not ready",
            bg="#f4f4f4",
            anchor="w"
        )
        self.colmap_status_label.pack(fill="x", pady=(0, 5))

        tk.Label(
            colmap_frame,
            text="Run COLMAP using frames and masks where available.",
            bg="#f4f4f4",
            anchor="w"
        ).pack(fill="x", pady=(0, 5))

        self.start_colmap_button = tk.Button(
            colmap_frame,
            text="Start 3D reconstruction",
            command=self.start_colmap_thread,
            width=25,
            height=2,
            state="disabled"
        )
        self.start_colmap_button.pack(pady=5)

        # Log section for displaying real-time logs of the experiment process.
        log_header = tk.Frame(right_panel, bg="#f4f4f4")
        log_header.pack(fill="x")

        tk.Label(
            log_header,
            text="Log",
            font=("Arial", 10, "bold"),
            bg="#f4f4f4"
        ).pack(side="left")

        clear_button = tk.Button(
            log_header,
            text="Clear",
            command=lambda: self.status_text.delete("1.0", tk.END)
        )
        clear_button.pack(side="right")

        self.status_text = tk.Text(
            right_panel,
            height=30,
            width=45,
            wrap="word"
        )
        self.status_text.pack(fill="both", expand=True, pady=(5, 0))

        def on_log_mousewheel(event):
            self.status_text.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_log_mousewheel(event):
            self.status_text.bind_all("<MouseWheel>", on_log_mousewheel)

        def unbind_log_mousewheel(event):
            self.status_text.unbind_all("<MouseWheel>")

        self.status_text.bind("<Enter>", bind_log_mousewheel)
        self.status_text.bind("<Leave>", unbind_log_mousewheel)

        # Information bar at the bottom of the GUI for displaying active time,
        # SAM 3 processing time, Photoshop time, and current experiment folder.
        bottom_bar = tk.Frame(main, bg="#d95b9f", height=45)
        bottom_bar.pack(fill="x", pady=(10, 0))
        bottom_bar.pack_propagate(False)

        self.active_time_label = tk.Label(
            bottom_bar,
            text="Active time: -",
            bg="#d95b9f",
            fg="white"
        )
        self.active_time_label.pack(side="left", expand=True)

        self.sam3_time_label = tk.Label(
            bottom_bar,
            text="SAM3 time: -",
            bg="#d95b9f",
            fg="white"
        )
        self.sam3_time_label.pack(side="left", expand=True)

        self.photoshop_time_label = tk.Label(
            bottom_bar,
            text="Photoshop time: -",
            bg="#d95b9f",
            fg="white"
        )
        self.photoshop_time_label.pack(side="left", expand=True)

        self.folder_label = tk.Label(
            bottom_bar,
            text="Experiment folder: -",
            bg="#d95b9f",
            fg="white"
        )
        self.folder_label.pack(side="left", expand=True)

    # This opens File explorer to select a video file and updates the label with the selected file name.
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("MP4 video files", "*.mp4"), # Only allows MP4 files to be selected.
            ]
        )
        # If a file is selected, update the video path and label, and reset the selected point.
        if file_path:
            self.video_path = Path(file_path)
            self.video_label.config(text=self.video_path.name)
            self.log(f"Selected video: {self.video_path}")

            self.selected_point = None
            self.point_label.config(text="Selected point: None")

    # Text prompt timer methods for starting and confirming the text prompt, which measures the active time spent on the text prompt.
    def start_text_prompt_timer(self):
        if self.prompt_type_entry.get() != "Text prompt":
            self.log("Error: Select Text prompt before starting text prompt timer.")
            return

        self.text_prompt_start_time = time.perf_counter()
        self.text_prompt_end_time = None
        self.text_prompt_confirmed = False
        self.active_prompt_duration = None

        self.active_time_label.config(text="Active time: -")
        self.log("Text prompt timer started.")

    def confirm_text_prompt(self):
        if self.prompt_type_entry.get() != "Text prompt":
            self.log("Error: Select Text prompt before confirming text prompt.")
            return

        text_prompt = self.text_prompt_entry.get().strip()

        if not text_prompt:
            self.log("Error: Text prompt cannot be empty.")
            return

        if self.text_prompt_start_time is None:
            self.log("Error: Start text prompt timer before confirming text prompt.")
            return

        self.text_prompt_end_time = time.perf_counter()
        self.active_prompt_duration = (
            self.text_prompt_end_time - self.text_prompt_start_time
        )
        self.active_prompt_duration = round(self.active_prompt_duration, 2)

        self.text_prompt_confirmed = True

        self.active_time_label.config(
            text=f"Active time: {self.active_prompt_duration:.2f} s"
        )

        self.log(
            f"Text prompt confirmed. Active prompt time: "
            f"{self.active_prompt_duration:.2f} seconds."
        )

    # Point prompt methods for starting point selection, displaying the first frame of the video,
    # and confirming the selected point, which measures the active time spent on point selection.
        if self.video_path is None:
            self.log("Error: Select a video before starting point selection.")
            return

        self.selected_point = None
        self.point_label.config(text="Selected point: None")

        self.point_selection_start_time = time.perf_counter()

        self.log("Point selection started.")
        self.log("Active user time measurement started.")

        self.show_first_frame()        

    def show_first_frame(self):
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            self.log("Error: Could not open video.")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.log("Error: Could not read the first frame of the video.")
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.original_height, self.original_width = frame.shape[:2]
        
        max_width = 600
        max_height = 350

        scale = min(
            max_width / self.original_width,
            max_height / self.original_height
        )

        self.display_width = int(self.original_width * scale)
        self.display_height = int(self.original_height * scale)

        image = Image.fromarray(frame)
        image = image.resize((self.display_width, self.display_height))

        self.tk_image = ImageTk.PhotoImage(image)

        self.frame_canvas.config(
            width=self.display_width,
            height=self.display_height
        )

        self.frame_canvas.delete("all")
        self.frame_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.log("Displayed the first frame of the video. Please click on the object you want to segment.")

    def on_frame_click(self, event):
        if self.original_width is None or self.original_height is None:
            self.log("Error: Original video dimensions are not set.")
            return

        real_x = int(event.x * self.original_width / self.display_width)
        real_y = int(event.y * self.original_height / self.display_height)

        self.selected_point = (real_x, real_y)

        self.frame_canvas.delete("point_marker")
        self.frame_canvas.create_oval(
            event.x - 5, event.y - 5, event.x + 5, event.y + 5,
            outline="red", tags="point_marker"
        )

        self.point_label.config(text=f"Selected point: {self.selected_point}")
        self.log(f"Selected point: {self.selected_point}")

    def confirm_point_selection(self):
        if self.selected_point is None:
            self.log("Error: Selected point: None")
            return

        self.point_selection_end_time = time.perf_counter()
        self.active_prompt_duration = (
            self.point_selection_end_time - self.point_selection_start_time
        )
        self.active_prompt_duration = round(self.active_prompt_duration, 2)

        self.log(
            f"Point selection confirmed. Active prompt time: "
            f"{self.active_prompt_duration:.2f} seconds."
        )
        
        self.active_time_label.config(
            text=f"Active time: {self.active_prompt_duration:.2f} s"
        )

    def start_experiment_thread(self):
        thread = threading.Thread(target=self.run_experiment)
        thread.daemon = True
        thread.start()

    # Experiment workflow method that validates user inputs, creates an experiment run, extracts frames, runs SAM3 processing, and creates datasets. 
    def run_experiment(self):
        prompt_type = self.prompt_type_entry.get()
        text_prompt = self.text_prompt_entry.get().strip()

        if prompt_type == "Point prompt" and self.active_prompt_duration is None:
            self.log("Error: Confirm point selection before running the experiment.")
            return

        if prompt_type == "Text prompt":
            if not text_prompt:
                self.log("Error: Text prompt cannot be empty.")
                return

            if not self.text_prompt_confirmed:
                self.log("Error: Confirm text prompt before running the experiment.")
                return

        if self.video_path is None:
            self.log("Error: No video selected.")
            return

        if self.video_path.suffix.lower() != ".mp4":
            self.log("Error: Only MP4 files are supported.")
            return

        interval = self.interval_entry.get()

        if prompt_type == "Point prompt" and self.selected_point is None:
            self.log("Error: Selected point: None")
            return

        try:
            interval = int(interval)
        except ValueError:
            self.log("Error: Subset interval must be a number.")
            return
        
        if interval <= 0:
            self.log("Error: Subset interval must be greater than 0.")
            return
        
        lighting = self.lighting_entry.get()
        background = self.background_entry.get()
        object_name = self.object_entry.get()
        object_structure = self.object_structure_entry.get()

        self.log("Starting experiment...")
        self.log(f"Video: {self.video_path}")
        self.log(f"Video lighting: {lighting}")
        self.log(f"Video background: {background}")
        self.log(f"Object: {object_name}")
        self.log(f"Selected point: {self.selected_point}")
        self.log(f"Subset interval: {interval}")
        self.log(f"Prompt type: {prompt_type}")

        if prompt_type == "Point prompt":
            self.log(f"Selected point: {self.selected_point}")
        else:
            self.log(f"Text prompt: {text_prompt}")

        try:
            # 1. Create experiment run
            manager = ExperimentManager()

            experiment = manager.create_experiment_run(
                video_path=self.video_path,
                description="Thesis experiment run",
            )

            paths = experiment["paths"]
            log_path = experiment["log_path"]

            self.current_metadata_path = experiment["metadata_path"]
            self.current_log_path = log_path

            experiment_root = Path(experiment["metadata_path"]).parent

            self.colmap_results_dir = experiment_root / "colmap_results"
            self.colmap_results_dir.mkdir(parents=True, exist_ok=True)  # Stores COLMAP results

            self.photoshop_frames_dir = (
                experiment_root / "datasets" / "photoshop_dataset" / "frames"
            )
            self.photoshop_masks_dir = (
                experiment_root / "datasets" / "photoshop_dataset" / "masks"
            )

            self.photoshop_frames_dir.mkdir(parents=True, exist_ok=True)
            self.photoshop_masks_dir.mkdir(parents=True, exist_ok=True)

            self.log(f"Experiment created with ID: {experiment['experiment_id']}")

            self.folder_label.config(
                text=f"Experiment folder: {experiment['experiment_id']}"
            )

            # 2. Extract frames
            manager.log(log_path, "Frame extraction started.")
            self.log("Frame extraction started.")

            extractor = FrameExtractor(
                video_path=self.video_path,
                output_dir=paths["frames"]
            )

            frame_metadata = extractor.extract_frames()

            manager.log(log_path, "Frame extraction completed.")
            self.log(f"Extracted {frame_metadata['video_total_frames']} frames from the video.")

            # 3. Run SAM3
            manager.log(log_path, "SAM3 processing started.")
            self.log("SAM3 processing started.")

            self.sam3_start_time = time.perf_counter()
            self.log("SAM3 timer started.")

            processor = SAM3Processor(
                video_path=self.video_path,
                output_dir=paths["sam3_masks"],
                log_callback=self.log
            )

            if prompt_type == "Point prompt":
                sam3_metadata = processor.process_video(
                    point=self.selected_point
                )
            else:
                sam3_metadata = processor.process_video(
                    text_prompt=text_prompt
                )

            self.sam3_end_time = time.perf_counter()

            manager.log(log_path, "SAM3 processing completed.")

            self.sam3_processing_duration = self.sam3_end_time - self.sam3_start_time
            self.sam3_processing_duration = round(self.sam3_processing_duration, 2)

            self.sam3_time_label.config(
                text=f"SAM3 time: {self.sam3_processing_duration:.2f} s"
            )

            self.log(
                f"SAM3 timer stopped. SAM3 processing time: "
                f"{self.sam3_processing_duration:.2f} seconds."
            )

            self.log(f"Saved {sam3_metadata['saved_masks']} masks.")

            # 4. Create datasets
            manager.log(log_path, "Dataset creation started.")
            self.log("Dataset creation started.")

            subset_creator = SubsetCreator(
                frames_dir=paths["frames"],
                original_dataset_dir=paths["original_dataset"],
                masks_dir=paths["sam3_masks"]
            )

            subset_metadata = subset_creator.create_datasets(interval=interval)

            manager.log(log_path, "Dataset creation completed.")

            self.log(
                f"Created datasets with {subset_metadata['selected_frames']} selected frames "
                f"and {subset_metadata['copied_masks']} SAM3 masks."
            )

            for frame_file in sorted(paths["original_dataset"].glob("*.png")):
                shutil.copy2(frame_file, self.photoshop_frames_dir / frame_file.name)

            self.log("Copied subset frames to Photoshop dataset frames folder.")

            self.photoshop_status_label.config(text="Status: Ready")
            self.photoshop_folder_label.config(text=f"Masks folder: {self.photoshop_masks_dir}")

            self.start_photoshop_button.config(state="normal")
            self.open_photoshop_folder_button.config(state="normal")

            self.log("Photoshop phase is ready.")
            self.log(f"Place Photoshop masks in: {self.photoshop_masks_dir}")

            # 5. Update metadata
            manager.log(log_path, "Metadata update started.")
            self.log("Metadata update started.")

            manager.update_metadata(
                metadata_path=experiment["metadata_path"],
                updates={
                    "capture_conditions": {
                        "lighting": lighting,
                        "background": background,
                        "object": object_name,
                        "object_structure": object_structure
                    },
                    "prompt": {
                        "type": prompt_type,
                        "point": {
                            "x": self.selected_point[0],
                            "y": self.selected_point[1],
                            "frame_index": 0,
                            "label": "foreground"
                        } if prompt_type == "Point prompt" else None,
                        "text": text_prompt if prompt_type == "Text prompt" else None
                    },
                    "photoshop_phase": {
                        "status": "ready",
                        "masks_folder": str(self.photoshop_masks_dir),
                        "time_seconds": self.photoshop_duration,
                        "validated": False
                    },
                    "subset_interval": interval,
                    "frame_extraction": frame_metadata,
                    "sam3_processing": {
                        **sam3_metadata,
                        "active_prompt_time_seconds": self.active_prompt_duration,
                        "processing_time_seconds": self.sam3_processing_duration
                    },
                    "dataset_creation": subset_metadata,
                    "dataset_status": {
                        "original_created": True,
                        "sam3_created": subset_metadata["copied_masks"] > 0,
                        "photoshop_created": False
                    },
                    "status": "datasets_created"
                }
            )

            manager.log(log_path, "Metadata updated.")
            self.log("Metadata updated.")

            manager.log(log_path, "Experiment run completed.")
            self.log("Experiment run completed.")

        # Catch any exceptions that occur during the experiment run and log them.
        except Exception as e:
            error_message = f"An error occurred: {e}"
            self.log(error_message)

            if "log_path" in locals():
                manager.log(log_path, error_message)

    # Photoshop timer methods for starting and stopping the Photoshop mask creation timer,
    # which measures the time spent creating masks in Photoshop.
    def start_photoshop_timer(self):
        
        if self.photoshop_masks_dir is None:
            self.log("Error: Photoshop masks folder is not ready.")
            return

        self.photoshop_start_time = time.perf_counter()
        self.photoshop_end_time = None
        self.photoshop_duration = None

        self.photoshop_status_label.config(text="Status: In progress")
        self.start_photoshop_button.config(state="disabled")
        self.stop_photoshop_button.config(state="normal")

        self.log("Photoshop timer started.")

        if self.current_log_path:
            manager = ExperimentManager()
            manager.log(
                self.current_log_path,
                "Photoshop mask creation started."
            )

    def stop_photoshop_timer(self):
        if self.photoshop_start_time is None:
            self.log("Error: Photoshop timer has not been started.")
            return

        self.photoshop_end_time = time.perf_counter()
        self.photoshop_duration = self.photoshop_end_time - self.photoshop_start_time
        self.photoshop_duration = round(self.photoshop_duration, 2)

        self.photoshop_status_label.config(text="Status: Timer stopped")
        self.photoshop_time_label.config(
            text=f"Photoshop time: {self.photoshop_duration:.2f} s"
        )

        self.stop_photoshop_button.config(state="disabled")
        self.validate_photoshop_button.config(state="normal")

        self.log(
            f"Photoshop timer stopped. Photoshop time: "
            f"{self.photoshop_duration:.2f} seconds."
        )
        if self.current_log_path:
            manager = ExperimentManager()
            manager.log(
                self.current_log_path,
                "Photoshop mask creation stopped."
            )

        if self.current_metadata_path is not None:
            manager = ExperimentManager()
            manager.update_metadata(
                metadata_path=self.current_metadata_path,
                updates={
                    "photoshop_phase": {
                        "status": "timer_stopped",
                        "masks_folder": str(self.photoshop_masks_dir),
                        "time_seconds": self.photoshop_duration,
                        "validated": False
                    }
                }
            )

            self.log("Photoshop timing saved to metadata.")
            if self.current_log_path:
                manager = ExperimentManager()
                manager.log(
                    self.current_log_path,
                    "Photoshop metadata saved."
                )

    # Opens the Photoshop target folder in the file explorer for the user to place the created masks.
    def open_photoshop_folder(self):
        if self.photoshop_masks_dir is None:
            self.log("Error: Photoshop masks folder is not ready.")
            return

        os.startfile(self.photoshop_masks_dir)

    # Validates the Photoshop masks by checking if the number of masks matches the number of original frames,
    # if the names of the masks match the names of the original frames, 
    # and if the resolution of the masks matches the resolution of the original frames.
    def validate_photoshop_masks(self):
        if self.photoshop_masks_dir is None:
            self.log("Error: Photoshop masks folder is not ready.")
            return

        original_dataset_dir = self.photoshop_masks_dir.parent.parent / "original_dataset"

        original_frames = sorted(original_dataset_dir.glob("*.png"))
        mask_files = sorted(self.photoshop_masks_dir.glob("*.png"))

        self.log(f"Expected masks: {len(original_frames)}")
        self.log(f"Found masks: {len(mask_files)}")

        if len(mask_files) != len(original_frames):
            self.photoshop_status_label.config(text="Status: Validation failed")
            self.log("Validation failed: mask count does not match original dataset frame count.")
            return
        
        frame_names = {frame.name for frame in original_frames}
        mask_names = {mask.name for mask in mask_files}

        missing_masks = frame_names - mask_names
        extra_masks = mask_names - frame_names

        if missing_masks:
            self.photoshop_status_label.config(text="Status: Validation failed")
            self.log(f"Validation failed: missing masks: {sorted(missing_masks)}")
            return

        if extra_masks:
            self.photoshop_status_label.config(text="Status: Validation failed")
            self.log(f"Validation failed: extra masks: {sorted(extra_masks)}")
            return

        for frame in original_frames:
            matching_mask = self.photoshop_masks_dir / frame.name

            frame_img = Image.open(frame)
            mask_img = Image.open(matching_mask)

            if frame_img.size != mask_img.size:
                self.photoshop_status_label.config(text="Status: Validation failed")
                self.log(
                    f"Validation failed: size mismatch in {frame.name} "
                    f"(frame: {frame_img.size}, mask: {mask_img.size})"
                )
                return
        
        self.photoshop_status_label.config(text="Status: Validation passed")
        self.log("Photoshop mask validation passed.")
        if self.current_log_path:
            manager = ExperimentManager()
            manager.log(
                self.current_log_path,
                "Photoshop phase completed."
            )

        # Update metadata to indicate that the Photoshop phase is completed and masks are validated
        # and ready for COLMAP reconstruction.
        self.colmap_status_label.config(text="Status: Ready")
        self.start_colmap_button.config(state="normal")
        self.log("COLMAP reconstruction is ready to start.")

    # Starts a new thread to run the COLMAP 3D reconstruction process for all datasets
    def start_colmap_thread(self):
        thread = threading.Thread(target=self.run_colmap_reconstruction)
        thread.daemon = True
        thread.start()

    # Runs COLMAP 3D reconstruction for all datasets and updates the metadata with the results.
    def run_colmap_reconstruction(self):
        if self.current_metadata_path is None:
            self.log("Error: No active experiment found.")
            return

        experiment_root = Path(self.current_metadata_path).parent
        datasets_dir = experiment_root / "datasets"
        colmap_results_dir = experiment_root / "colmap_results"

        dataset_configs = {
            "original": {
                "image_dir": datasets_dir / "original_dataset",
                "mask_dir": None
            },
            "sam3": {
                "image_dir": datasets_dir / "sam3_dataset" / "frames",
                "mask_dir": datasets_dir / "sam3_dataset" / "masks"
            },
            "photoshop": {
                "image_dir": datasets_dir / "photoshop_dataset" / "frames",
                "mask_dir": datasets_dir / "photoshop_dataset" / "masks"
            }
        }

        self.start_colmap_button.config(state="disabled")
        self.colmap_status_label.config(text="Status: Running")
        self.log("COLMAP 3D reconstruction started.")
        if self.current_log_path:
            manager = ExperimentManager()
            manager.log(
                self.current_log_path,
                "COLMAP 3D reconstruction started."
            )

        self.colmap_start_time = time.perf_counter()
        colmap_results = {}

        colmap = COLMAPProcessor()

        for dataset_name, config in dataset_configs.items():
            image_dir = config["image_dir"]
            mask_dir = config["mask_dir"]
            output_dir = colmap_results_dir / dataset_name

            self.log(f"Running COLMAP for {dataset_name} dataset...")
            self.log(f"Images: {image_dir}")

            if mask_dir is not None:
                self.log(f"Masks: {mask_dir}")
            else:
                self.log("Masks: not used")

            result = colmap.run_sparse_reconstruction(
                dataset_name=dataset_name,
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_dir=output_dir
            )

            colmap_results[dataset_name] = result

            if result["success"]:
                self.log(f"COLMAP completed for {dataset_name}.")
                self.log(f"Registered images: {result['registered_images']}")
                self.log(f"3D points: {result['points3D']}")
                self.log(f"Mean reprojection error: {result['mean_reprojection_error']}")
                self.log(f"PLY model: {result['ply_model_path']}")
            else:
                self.log(f"COLMAP failed for {dataset_name}.")
                self.log(f"Error: {result['error']}")

        self.colmap_end_time = time.perf_counter()
        self.colmap_duration = round(
            self.colmap_end_time - self.colmap_start_time,
            2
        )

        if all(result["success"] for result in colmap_results.values()):
            colmap_status = "completed"
            gui_status = "Status: Completed"
        elif any(result["success"] for result in colmap_results.values()):
            colmap_status = "completed_with_errors"
            gui_status = "Status: Completed with errors"
        else:
            colmap_status = "failed"
            gui_status = "Status: Failed"

        self.colmap_status_label.config(text=gui_status)

        self.log(
            f"COLMAP reconstruction finished in "
            f"{self.colmap_duration:.2f} seconds."
        )
        if self.current_log_path:
            manager = ExperimentManager()
            manager.log(
                self.current_log_path,
                "COLMAP 3D reconstruction completed."
            )

        manager = ExperimentManager()
        manager.update_metadata(
            metadata_path=self.current_metadata_path,
            updates={
                "colmap_reconstruction": {
                    "status": colmap_status,
                    "total_time_seconds": self.colmap_duration,
                    "results_folder": str(colmap_results_dir),
                    "results": colmap_results
                },
                "status": "colmap_completed"
                if colmap_status != "failed"
                else "colmap_failed"
            }
        )

        self.log("COLMAP metadata saved.")
        self.start_colmap_button.config(state="normal")
        if self.current_log_path:
            manager = ExperimentManager()
            manager.log(
                self.current_log_path,
                "COLMAP metadata saved."
            )

    # Logs messages to the status text area in the GUI and ensures that the latest log entry is visible.
    def log(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()        

# Main entry point for the GUI application. Initializes the Tkinter root window and starts the main event loop.
if __name__ == "__main__":
    root = tk.Tk()
    app = ThesisApp(root)
    root.mainloop()