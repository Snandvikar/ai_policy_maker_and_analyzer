"""
API_Key Management
Handles storing and retrieving API_Keys for NDAP (ndap.niti.gov.in)

How to get your API_Key:
  1. Go to https://ndap.niti.gov.in  →  sign in (free account)
  2. Open any dataset, e.g. https://ndap.niti.gov.in/get-api/7973
  3. Click the blue "Copy API" button on that page
     – OR –
     Open DevTools (F12) → Network → type "openapi" → copy the URL
     The key is the value of the "API_Key=" query parameter
  4. Paste it when prompted, or set env var  NDAP_API_KEY=gAAAAA...
"""

import os
from pathlib import Path

# .api_key file lives next to this script
API_KEY_FILE = Path(__file__).parent / ".api_key"


def save_api_key(api_key: str) -> None:
    """Persist API_Key to a local hidden file so it is reused across runs."""
    with open(API_KEY_FILE, "w") as f:
        f.write(api_key.strip())
    print(f"✓ API_Key saved to {API_KEY_FILE}")


def load_api_key_from_file() :
    """Return the cached API_Key, or None if the file is missing / empty."""
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text().strip()
        return key or None
    return None


def load_api_key_from_env() :
    """Return the key from the NDAP_API_KEY environment variable."""
    return os.getenv("NDAP_API_KEY")


def get_api_key(force_input: bool = False) :
    """
    Return the API_Key using this priority:
      1. Environment variable  NDAP_API_KEY
      2. Cached file           .api_key   (skipped when force_input=True)
      3. Interactive prompt    (only when force_input=True or no other source)
    """
    # 1 – environment variable
    key = load_api_key_from_env()
    if key:
        print("✓ Loaded API_Key from NDAP_API_KEY environment variable")
        return key

    # 2 – cached file
    if not force_input:
        key = load_api_key_from_file()
        if key:
            print(f"✓ Loaded API_Key from cache ({API_KEY_FILE})")
            return key

    # 3 – interactive prompt
    print("\n" + "=" * 70)
    print("NDAP API_Key Required")
    print("=" * 70)
    print("""
How to get your API_Key (takes ~1 minute):
  1. Open https://ndap.niti.gov.in  and sign in (free account)
  2. Navigate to any dataset's API page, e.g.:
       https://ndap.niti.gov.in/get-api/7973
  3. Click the blue  "Copy API"  button  – this copies the full API URL.
     The API_Key is the long  gAAAAAB...  string after  API_Key=  in that URL.
  4. Paste just the key value below (without "API_Key=").

Alternative – set an environment variable and re-run:
  Windows : $env:NDAP_API_KEY = "gAAAAAB..."
  Linux   : export NDAP_API_KEY="gAAAAAB..."
""")
    key = input("Paste your API_Key here: ").strip()

    if key:
        save_api_key(key)
        return key

    return None


# ── standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("NDAP API_Key Manager")
    print("=" * 70)
    api_key = get_api_key(force_input=False)
    if api_key:
        print(f"\n✓ API_Key loaded : {api_key[:40]}...")
        print(f"✓ Cached at      : {API_KEY_FILE}")
    else:
        print("\n✗ No API_Key available")