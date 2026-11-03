"""Reassemble architecture_unet.h5 from its parts and verify its SHA-256.

GitHub limits files to 100 MB, so the 373 MB architecture U-Net is stored as
architecture_unet.h5.part01 ... part04. Run this script once after cloning or
downloading the repository:

    python models/join_architecture_unet.py
"""

import hashlib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
TARGET = MODELS_DIR / "architecture_unet.h5"
PART_PATTERN = "architecture_unet.h5.part*"
EXPECTED_PARTS = 4
EXPECTED_SHA256 = "4bc19bac1439f984c56f6f28f2e5ae3eb0e00b3790e1c525efe7ddb5eda1bfc5"
READ_BYTES = 1 << 20


def main() -> None:
    parts = sorted(MODELS_DIR.glob(PART_PATTERN))
    if len(parts) != EXPECTED_PARTS:
        raise SystemExit(f"expected {EXPECTED_PARTS} parts in {MODELS_DIR}, found {len(parts)}")

    digest = hashlib.sha256()
    with TARGET.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                while chunk := src.read(READ_BYTES):
                    out.write(chunk)
                    digest.update(chunk)

    if digest.hexdigest() != EXPECTED_SHA256:
        TARGET.unlink()
        raise SystemExit("SHA-256 mismatch: the parts are incomplete or corrupted; re-download them")
    print(f"wrote {TARGET} ({TARGET.stat().st_size:,} bytes), SHA-256 verified")


if __name__ == "__main__":
    main()
