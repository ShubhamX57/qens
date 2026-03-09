import glob
import os
import pandas as pd

def build_metadata(folder_path: str) -> pd.DataFrame:
    records = []
    for file in glob.glob(os.path.join(folder_path, "*.nxspe")):
        name = os.path.basename(file).replace(".nxspe", "")
        if name.startswith("empty"):
            continue

        parts = name.split("_")
        if len(parts) < 4:
            continue

        try:
            sample = parts[0]
            temperature = float(parts[1])
            Ei = float(parts[2])
            scattering = parts[3]

            records.append(
                {
                    "filepath": file,
                    "sample": sample,
                    "temperature": temperature,
                    "Ei": Ei,
                    "scattering": scattering,
                }
            )
        except ValueError:
            continue

    return pd.DataFrame(records)