from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import certifi
import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "dgs3mo.csv"


def main():
    response = requests.get(URL, timeout=30, verify=certifi.where())
    response.raise_for_status()

    data = pd.read_csv(BytesIO(response.content))

    required_columns = {"observation_date", "DGS3MO"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Unexpected FRED columns: {list(data.columns)}")

    data["DGS3MO"] = pd.to_numeric(data["DGS3MO"], errors="coerce")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)

    print(f"Source: {URL}")
    print(f"Retrieved (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Observations: {len(data)}")
    print("Date range:", data["observation_date"].min(), "to", data["observation_date"].max())
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
