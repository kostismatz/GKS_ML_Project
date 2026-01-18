import pandas as pd
from urbansound.paths import METADATA_CSV, AUDIO_DIR

CLASS_ID_TO_NAME = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}


def load_metadata() -> pd.DataFrame:

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata CSV: {METADATA_CSV}\n"
            "Expected dataset at: data/raw/UrbanSound8K/\n"
            "with metadata/UrbanSound8K.csv and audio/fold*/...wav"
        )

    df = pd.read_csv(METADATA_CSV)

    # set paths for all specific files:
    df["filepath"] = df.apply(
        lambda r: str(AUDIO_DIR / f"fold{int(r['fold'])}" / r["slice_file_name"]),
        axis=1,
    )

    df["class_name"] = df["classID"].map(CLASS_ID_TO_NAME)

    return df
