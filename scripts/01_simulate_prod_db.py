import pathlib
import urllib.request
import pandas as pd
import duckdb


def main() -> None:
    DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASE_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/"
    FILES_TO_DOWNLOAD = {
        "airports": "airports.dat",
        "airlines": "airlines.dat",
        "routes": "routes.dat",
        "countries": "countries.dat",
        "planes": "planes.dat",
    }
    COLUMN_NAMES = {
        "airports": [
            "airport_id",
            "name",
            "city",
            "country",
            "iata",
            "icao",
            "latitude",
            "longitude",
            "altitude",
            "timezone",
            "dst",
            "tz_db",
            "type",
            "source",
        ],
        "airlines": [
            "airline_id",
            "name",
            "alias",
            "iata",
            "icao",
            "callsign",
            "country",
            "active",
        ],
        "routes": [
            "airline",
            "airline_id",
            "source_airport",
            "source_airport_id",
            "destination_airport",
            "destination_airport_id",
            "codeshare",
            "stops",
            "equipment",
        ],
        "countries": ["name", "iso_code", "dafif_code"],
        "planes": ["name", "iata", "icao"],
    }
    PROD_DB_PATH = str(DATA_DIR / "prod_openflights.db")

    with duckdb.connect(PROD_DB_PATH) as con:
        for name, filename in FILES_TO_DOWNLOAD.items():
            url = f"{BASE_URL}{filename}"
            path = DATA_DIR / filename
            if not path.exists():
                urllib.request.urlretrieve(url, path)
                print(f"Downloaded: {path}")
            df = pd.read_csv(  # noqa: F841
                path, header=None, names=COLUMN_NAMES[name], na_values=["\\N"]
            )
            con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM df')
        print(f"\n'Production' database simulated at: {PROD_DB_PATH}")
        print("Tables created:", con.sql("SHOW TABLES;").fetchall())


if __name__ == "__main__":
    main()
