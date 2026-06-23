import os
import numpy as np
import matplotlib.pyplot as plt
from model.io import (
    load_rcm_product, 
    run_pred_model,
    scale_hh_hv, 
    build_land_masks,
    normalize_and_prepare_images,
    resource_path,
)

# Cycle through all scenes in a folder and generate predictions for each scene
def get_pred_map_bulk(folder_path, model_path, progress=None):
    scene_folders = [
        os.path.join(folder_path, d)
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    # Ignore prediction folder if it exists
    scene_folders = [d for d in scene_folders if os.path.basename(d) != "predictions"]

    # Save predictions to a folder as png files
    pred_save_path = os.path.join(folder_path, "predictions")
    os.makedirs(pred_save_path, exist_ok=True)

    model_pred_save_path = os.path.join(pred_save_path, os.path.split(model_path)[-1])
    os.makedirs(model_pred_save_path, exist_ok=True)

    for i, scene_folder in enumerate(scene_folders):
        print(f"{i} / {len(scene_folders)} - Processing scene {i + 1}/{len(scene_folders)}: {scene_folder}")

        rcm_data = load_rcm_product(scene_folder)
        if rcm_data is None:
            print(f"Unable to load the selected RCM product for scene: {scene_folder}. Skipping.")
            continue

        rcm_data["sar_img"] = None

        rcm_200m_data = scale_hh_hv(
            rcm_data,
            target_spacing_m=200,
        )

        land_mask = build_land_masks(rcm_200m_data)

        raw_img, orig_img, hist, n_valid, nan_mask, geo_coord_helpers = (
            normalize_and_prepare_images(rcm_200m_data, "mean-std")
        )

        prediction_vars = run_pred_model(
            "Model",
            rcm_200m_data,
            land_mask,
            model_path,
            target_width=rcm_200m_data["dst_width"],
            target_height=rcm_200m_data["dst_height"],
            target_spacing=200,
            device="cpu",
        )

        pred = prediction_vars[0][1]

        # Display the prediction map
        # plt.figure(figsize=(10, 10))
        # plt.imshow(pred, cmap="viridis")
        # plt.title("Prediction Map")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.show()

        # Save the prediction map as a PNG file
        pred_filename = os.path.join(model_pred_save_path, f"{os.path.split(scene_folder)[-1]}_pred.png")
        plt.imsave(pred_filename, pred, cmap="viridis")

def main():
    folder_path = input("Enter the path to the folder containing RCM scenes: ")
    model_path = "model\\prediction_model\\best_mFscore_iter_3500.pth"  # Replace with the path
    get_pred_map_bulk(folder_path, model_path)

if __name__ == "__main__":
    main()