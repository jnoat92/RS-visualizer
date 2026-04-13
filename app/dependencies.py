from dataclasses import dataclass
import customtkinter as ctk
from tkinter import Canvas

@dataclass
class AppDeps:
    app: ctk.CTk
    app_state: object

    canvas: Canvas
    sidebar: ctk.CTkFrame
    minimap: object
    minimap_window_id: int
    status_bar: object

    setup_window: object

    annotation_panel: object
    evaluation_panel: object
    annotation_window: object
    evaluation_window: object

    loading_bar: ctk.CTkProgressBar
    loading_bar_label: ctk.CTkLabel

    widgets: dict
