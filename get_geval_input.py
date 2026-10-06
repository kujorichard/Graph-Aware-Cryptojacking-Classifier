import json
from pathlib import Path
import shutil

folder_path = Path("graph_aware_classifier/batch_detection_output/")
dest_path = Path("geval/geval_batch_runner/input_files/")

total = 0
for item in folder_path.iterdir():
    if item.is_file() and item.suffix == ".json":
        with open(item, "r") as f:
            data = json.load(f)

            if data["random_forest"]["predicted_class"] == 1:
                print(f"File Name: {item.name}")
                shutil.copy(item, dest_path)
                total+=1

print(f"Total number of files with predicted_class 1: {total}")