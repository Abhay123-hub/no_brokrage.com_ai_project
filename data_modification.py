import pandas as pd
import re
from typing import Optional

def clean_price(value) -> Optional[float]:
    """
    Normalize price strings to a float representing Indian rupees (INR).
    Rules:
      - 'cr' or 'crore' -> multiply by 1e7 (1 crore = 10,000,000)
      - 'lakh' or 'lac' -> multiply by 1e5 (1 lakh = 100,000)
      - handles ranges like '50-60 lakh' -> returns the average (in INR)
      - handles formatted numbers like '1,50,00,000' or '₹1,50,00,000'
      - returns numeric values unchanged (int/float)
      - returns None if no numeric info found
    """
    if pd.isna(value):
        return None

    # If already numeric
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None

    s = str(value).strip().lower()
    # remove currency symbol and common separators
    s = s.replace("₹", "").replace(",", "").replace("inr", "").strip()

    # remove '/sqft', 'psf', 'per sqft', etc. — we only keep numeric part and unit words
    s = re.sub(r"/?per\s*sq(?:\.| )?ft|psf|/sqft", "", s)

    # handle something like 'approx' or 'around'
    s = re.sub(r"\b(approx|around|~|estimated|est\.?)\b", "", s).strip()

    # if there's a range like '50-60 lakh' or '50 to 60 lakh'
    range_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|to)\s*([0-9]+(?:\.[0-9]+)?)", s)
    if range_match:
        a = float(range_match.group(1))
        b = float(range_match.group(2))
        num = (a + b) / 2.0
        # decide unit by checking rest of string
        if "cr" in s or "crore" in s:
            return num * 1e7
        if "lakh" in s or "lac" in s:
            return num * 1e5
        return num  # no explicit unit

    # find the first numeric token
    num_match = re.search(r"[0-9]+(?:\.[0-9]+)?", s)
    if not num_match:
        return None

    num = float(num_match.group(0))

    # decide multiplier
    if "cr" in s or "crore" in s:
        return num * 1e7
    if "lakh" in s or "lac" in s:
        return num * 1e5

    # if string had implicit large separators (e.g., 15000000) we already stripped commas,
    # so the numeric value could already be absolute rupees
    return num


class Data:
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path

    def read_csv_safe(self, filename: str) -> pd.DataFrame:
        path = f"{self.base_path}/{filename}"
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            raise
        except Exception as e:
            # surface a helpful error
            raise RuntimeError(f"Failed to read {path}: {e}")

    def get_data(self) -> pd.DataFrame:
        projects = self.read_csv_safe("project.csv")
        addresses = self.read_csv_safe("ProjectAddress.csv")
        configs = self.read_csv_safe("ProjectConfiguration.csv")
        variants = self.read_csv_safe("ProjectConfigurationVariant.csv")

        # Merge with explicit suffixes
        merged1 = pd.merge(
            projects,
            addresses,
            left_on="id",
            right_on="projectId",
            how="left",
            suffixes=("", "_address"),
            validate="one_to_many"  # or adjust depending on cardinality
        )

        merged2 = pd.merge(
            merged1,
            configs,
            left_on="id",
            right_on="projectId",
            how="left",
            suffixes=("", "_config"),
            validate="one_to_many"
        )

        # After the above merge, "id" still refers to projects.id; config rows may have their own id column
        # Many CSV exports name config id as 'id' too; to be safe, identify the configuration id column:
        # Prefer explicit configuration id column name if present, else try 'id_config' created by suffixing
        config_id_col = None
        if "id_config" in merged2.columns:
            config_id_col = "id_config"
        elif "id_config" not in merged2.columns and "id_y" in merged2.columns:
            config_id_col = "id_y"
        elif "configurationId" in variants.columns:
            # we'll merge using project configuration's id -> configurationId
            # but left_on must match a col that represents configuration id in merged2
            # try to find candidate by checking common names
            for cand in ("id_config", "configurationId", "configId", "id_y"):
                if cand in merged2.columns:
                    config_id_col = cand
                    break

        if config_id_col is None:
            # fallback: try to merge on projectId / configurationId if that makes sense
            master_df = pd.merge(merged2, variants, left_on="id", right_on="configurationId", how="left")
        else:
            master_df = pd.merge(merged2, variants, left_on=config_id_col, right_on="configurationId", how="left", suffixes=("", "_variant"))

        # choose safe column selection: only keep columns that exist
        desired_cols = [
            "projectName","status","possessionDate","fullAddress","pincode","type",
            "carpetArea","price","bathrooms","balcony","listingType","furnishedType",
            "projectCategory","projectType","propertyCategory","landmark","cityId",
            "localityId","subLocalityId","propertyImages","floorPlanImage"
        ]
        existing = [c for c in desired_cols if c in master_df.columns]
        chatbot_df = master_df[existing].copy()

        return chatbot_df

    def clean_data(self) -> pd.DataFrame:
        chatbot_df = self.get_data()

        # apply clean_price once and safely
        if "price" in chatbot_df.columns:
            chatbot_df["price"] = chatbot_df["price"].apply(clean_price)

        # Convert carpet area to numeric (coerce bad values to NaN)
        if "carpetArea" in chatbot_df.columns:
            chatbot_df["carpetArea"] = pd.to_numeric(chatbot_df["carpetArea"], errors="coerce")

        # Fill missing values with blank or defaults where appropriate
        if "status" in chatbot_df.columns:
            chatbot_df["status"] = chatbot_df["status"].fillna("UNKNOWN")

        if "listingType" in chatbot_df.columns:
            chatbot_df["listingType"] = chatbot_df["listingType"].fillna("Sell")
        chatbot_df.to_csv("cleaned_master_for_chatbot.csv", index=False)

        # Reset index and return
        return chatbot_df.reset_index(drop=True)



