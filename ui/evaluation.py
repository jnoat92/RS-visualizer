'''
Evaluation panel setup and functions

Last modified: Jan 2026
'''

import customtkinter as ctk
from tkinter import messagebox
import json
import os
import datetime
import csv

class EvaluationPanel(ctk.CTkFrame):
    def __init__(self, parent, command_parent=None):
        super().__init__(parent)

        if command_parent is None:
            command_parent = self
        self.command_parent = command_parent
        
        self.scene_name = ""
        self.unsaved_changes = False

        self.region_evaluation = ctk.StringVar(value=" ")
        self.boundary_evaluation = ctk.StringVar(value=" ")

        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        ctk.CTkLabel(self, text="").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(self, text="Region:").grid(row=1, column=0, sticky="w")
        ctk.CTkLabel(self, text="Boundaries:").grid(row=2, column=0, sticky="w")

        # Region Radiobuttons
        ctk.CTkRadioButton(self, text="Highly accurate", variable=self.region_evaluation,
                           value="high").grid(row=1, column=1, sticky="nsew", padx=(0,5))
        ctk.CTkRadioButton(self, text="Sufficient", variable=self.region_evaluation,
                           value="sufficient").grid(row=1, column=2, sticky="nsew", padx=(0,5))
        ctk.CTkRadioButton(self, text="Not sufficient", variable=self.region_evaluation,
                           value="not sufficient").grid(row=1, column=3, sticky="nsew", padx=(0,5))

        # Boundary Radiobuttons
        ctk.CTkLabel(self, text="Operational Capability").grid(row=0, column=1, columnspan=3, sticky="nsew")
        ctk.CTkRadioButton(self, text="Highly accurate", variable=self.boundary_evaluation,
                           value="high").grid(row=2, column=1, sticky="nsew", padx=(0,5))
        ctk.CTkRadioButton(self, text="Sufficient", variable=self.boundary_evaluation,
                           value="sufficient").grid(row=2, column=2, sticky="nsew", padx=(0,5))
        ctk.CTkRadioButton(self, text="Not sufficient", variable=self.boundary_evaluation,
                           value="not sufficient").grid(row=2, column=3, sticky="nsew", padx=(0,5))

        # Notes
        ctk.CTkLabel(self, text="Notes:").grid(row=0, column=4, sticky="w", padx=10)
        self.notes_text = ctk.CTkTextbox(self, width=300, height=100)
        self.notes_text.grid(row=1, column=4, rowspan=3, pady=(10, 0), padx=10)

        self.save_button = ctk.CTkButton(self, text="Save Evaluation", command=self.save_evaluation)
        self.save_button.grid(row=0, column=0, pady=10)

        self._trace_region_id = self.region_evaluation.trace_add("write", lambda *args: self._mark_unsaved())
        self._trace_boundary_id = self.boundary_evaluation.trace_add("write", lambda *args: self._mark_unsaved())
        self.notes_text.bind("<KeyRelease>", lambda e: self._mark_unsaved())

    def _mark_unsaved(self):
        if not self.unsaved_changes:
            self.unsaved_changes = True
            self.save_button.configure(text="* Save Evaluation")

    def set_scene_name(self, name):
        self.scene_name = name
        self.after(100, self.load_existing_evaluation)

    def load_existing_evaluation(self):
        filename = os.path.join("evaluations", "evaluation_data.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                    if self.scene_name in data:
                        scene_data = data[self.scene_name]

                        lbl_source = self.command_parent.mode_var_lbl_source.get()
                        if lbl_source in scene_data:

                            self.region_evaluation.set(scene_data[lbl_source].get("region", "").strip())
                            self.boundary_evaluation.set(scene_data[lbl_source].get("boundaries", "").strip())

                            self.notes_text.configure(state=ctk.NORMAL)
                            self.notes_text.delete("1.0", "end")
                            self.notes_text.insert("end", scene_data[lbl_source].get("notes", ""))

                            self.unsaved_changes = False
                            self.save_button.configure(text="Save Evaluation")
                            messagebox.showinfo("Loaded", f"Existing evaluation loaded for prediction '{lbl_source}' on scene '{self.scene_name}'", parent=self.master)
                        else:
                            self.reset_fields()
                    else:
                        self.reset_fields()
                except json.JSONDecodeError:
                    pass
        else:
            self.reset_fields()

    def save_evaluation(self):
        if not self.scene_name:
            messagebox.showerror("Error", "Scene name is not set.")
            return False

        lbl_source = self.command_parent.mode_var_lbl_source.get()
        if lbl_source == "Custom_Annotation":
            messagebox.showerror("Warning", "Please, evaluate segmentation source different from 'Custom Annotation'.")
            return False

        region_accuracy = self.region_evaluation.get().strip()
        boundary_accuracy = self.boundary_evaluation.get().strip()
        notes = self.notes_text.get("1.0", "end").strip()

        if region_accuracy == '' and boundary_accuracy == '':
            messagebox.showerror("Warning", "Evaluation incomplete")
            return False
        if region_accuracy == '':
            messagebox.showerror("Warning", "Evaluation incomplete for regions")
            return False
        if boundary_accuracy == '':
            messagebox.showerror("Warning", "Evaluation incomplete for boundaries")
            return False

        new_data = {
            self.scene_name: {
                lbl_source:{
                    "region": region_accuracy,
                    "boundaries": boundary_accuracy,
                    "notes": notes
                }
            }
        }

        os.makedirs("evaluations", exist_ok=True)
        filename = os.path.join("evaluations", "evaluation_data.json")

        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        overwrite = False
        if self.scene_name in existing_data:
            if lbl_source in existing_data[self.scene_name]:
                overwrite = messagebox.askyesno("Overwrite?",
                    f"An evaluation already exists for '{lbl_source}' on scene '{self.scene_name}'. Do you want to replace it?",
                    parent=self.master)
                if not overwrite:
                    return False
            existing_data[self.scene_name].update(new_data[self.scene_name])
        else:
            existing_data.update(new_data)

        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=4)

        self._save_to_csv(self.scene_name, lbl_source, region_accuracy, boundary_accuracy, notes)

        self.unsaved_changes = False
        self.save_button.configure(text="Save Evaluation")
        # self._show_silent_popup("Saved" if not overwrite else "Updated", f"Evaluation saved to {filename}")
        messagebox.showinfo("Saved" if not overwrite else "Updated", f"Evaluation saved to {filename}", parent=self.master)
        # self.status_label.configure(text="Evaluation saved to {filename}" if not overwrite else "Evaluation updated to {filename}")
        # self.after(3000, lambda: self.status_label.configure(text=""))
        return True

    def _save_to_csv(self, scene_name, lbl_source, region_eval, boundary_eval, notes):
        csv_file = os.path.join("evaluations", "evaluation_data.csv")
        file_exists = os.path.isfile(csv_file)
        fieldnames = ["scene_name", "lbl_source", "region", "boundaries", "notes", "timestamp"]

        rows = []
        if file_exists:
            with open(csv_file, "r", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        rows = [row for row in rows if row["scene_name"] != scene_name or row["lbl_source"] != lbl_source]

        new_row = {
            "scene_name": scene_name,
            "lbl_source": lbl_source,
            "region": region_eval,
            "boundaries": boundary_eval,
            "notes": notes,
            "timestamp": datetime.datetime.now().isoformat()
        }
        rows.append(new_row)

        with open(csv_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


    def _show_silent_popup(self, title, message):
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        ctk.CTkLabel(popup, text=message, padx=20, pady=10).pack()
        ctk.CTkButton(popup, text="OK", command=popup.destroy).pack(pady=(0, 10))

        popup.update_idletasks()
        w = popup.winfo_width()
        h = popup.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")
        popup.wait_window()


    def set_enabled(self, enabled=True):
        state = ctk.NORMAL if enabled else ctk.DISABLED
        for child in self.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass

    def reset_fields(self):

        self.region_evaluation.trace_remove("write", self._trace_region_id)
        self.boundary_evaluation.trace_remove("write", self._trace_boundary_id)

        self.region_evaluation.set(" ")
        self.boundary_evaluation.set(" ")

        self._trace_region_id = self.region_evaluation.trace_add("write", lambda *args: self._mark_unsaved())
        self._trace_boundary_id = self.boundary_evaluation.trace_add("write", lambda *args: self._mark_unsaved())

        self.notes_text.configure(state=ctk.NORMAL)
        self.notes_text.delete("1.0", "end")

        self.unsaved_changes = False
        self.save_button.configure(text="Save Evaluation")


# if __name__ == '__main__':
#     ctk.set_appearance_mode("System")  # "Dark", "Light", or "System"
#     ctk.set_default_color_theme("blue")  # or "green", "dark-blue", etc.

#     root = ctk.CTk()
#     root.title("Evaluation Panel")

#     panel = EvaluationPanel(root)
#     panel.pack(padx=10, pady=10, fill="both", expand=True)

#     panel.set_scene_name("example_scene")

#     root.mainloop()
