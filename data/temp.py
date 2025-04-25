from pathlib import Path
import pandas as pd
import shutil
from PIL import Image


def copy_relevant_images():
    csv_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\BMv3\labels\v3_mgs_01.csv")
    img_base_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\BMv3")
    save_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\MGS_data")
    save_path.mkdir(exist_ok=True)

    mgs_data = pd.read_csv(csv_path, sep=",")
    img_names = mgs_data.get("index").tolist()

    allowed_extensions = [".jpg", ".jpeg", ".png"]

    for img_name in img_names:
        img_stem = Path(img_name).stem  # Nur der Name ohne Extension (z.B. "mr_DSC_3771")

        # Suche das Bild mit beliebiger Extension
        matching_files = list(img_base_path.glob(f"{img_stem}.*"))

        matching_files = [f for f in matching_files if f.suffix.lower() in allowed_extensions]

        if not matching_files:
            print(f"No matching file found for {img_stem}. Skipping.")
            continue

        # Nimm das erste gefundene passende Bild
        img_path = matching_files[0]
        save_img_path = save_path / img_path.name

        if not save_img_path.exists():
            shutil.copy(img_path, save_img_path)
            print(f"Copied {img_path} -> {save_img_path}")
        else:
            print(f"{save_img_path} already exists.")
    print("Copying images done.")

def copy_relevant_csv_data():
    mgs_csv_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\BMv3\labels\v3_mgs_01.csv")
    main_csv_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\BMv3\labels\v3_main.csv")
    output_label_dir = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\MGS_data\labels")
    output_label_dir.mkdir(exist_ok=True)

    # Main CSV einlesen
    main_df = pd.read_csv(main_csv_path, sep=",")

    # Bildnamen im Verzeichnis holen (ohne Extension)
    all_img_paths = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\MGS_data").glob("*.jpg")
    all_img_names = [img_path.stem for img_path in all_img_paths]

    # Filter: Zeilen, bei denen 'index' (ohne Extension) in den Bildnamen ist
    main_df['index_stem'] = main_df['index'].apply(lambda x: Path(x).stem)
    filtered_df = main_df[main_df['index_stem'].isin(all_img_names)].drop(columns=['index_stem'])

    # CSVs speichern
    shutil.copy(mgs_csv_path, output_label_dir / "v3_mgs_01.csv")
    filtered_df.to_csv(output_label_dir / "v3_main.csv", sep=",", index=False)

    print("Copying csv data done.")
    
if __name__ == "__main__":
    copy_relevant_csv_data()