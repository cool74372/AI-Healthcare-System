import os
import shutil

BASE = "data/chest_xray"

def merge_folders(src, dst):
    for category in ["NORMAL", "PNEUMONIA"]:
        src_path = os.path.join(src, category)
        dst_path = os.path.join(dst, category)

        if not os.path.exists(src_path):
            continue

        os.makedirs(dst_path, exist_ok=True)

        for file in os.listdir(src_path):
            src_file = os.path.join(src_path, file)
            dst_file = os.path.join(dst_path, file)

            # Avoid overwrite
            if not os.path.exists(dst_file):
                shutil.move(src_file, dst_file)

def process():
    nested_path = os.path.join(BASE, "chest_xray")

    if not os.path.exists(nested_path):
        print("No nested folder found. You're good!")
        return

    print("🔄 Merging nested dataset...")

    for split in ["train", "test", "val"]:
        src = os.path.join(nested_path, split)
        dst = os.path.join(BASE, split)

        if os.path.exists(src) and os.path.exists(dst):
            merge_folders(src, dst)

    # Remove nested folder after merging
    shutil.rmtree(nested_path)

    print("✅ Merge complete. Dataset is clean now!")

if __name__ == "__main__":
    process()