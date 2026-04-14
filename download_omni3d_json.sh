#!/usr/bin/env bash
set -euo pipefail

# Download Omni3D_json.zip and extract it.
# Usage:
#   ./download_omni3d_json.sh [target_dir]
# Example:
#   ./download_omni3d_json.sh ~/datasets/omni3d

URL="https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip"
TARGET_DIR="${1:-$PWD/data/omni3d_download}"
ARCHIVE_PATH="$TARGET_DIR/Omni3D_json.zip"

mkdir -p "$TARGET_DIR"

echo "Downloading Omni3D annotations archive to: $ARCHIVE_PATH"
# -c resumes partial downloads if interrupted.
wget -c "$URL" -O "$ARCHIVE_PATH"

echo "Testing archive integrity..."
unzip -t "$ARCHIVE_PATH" >/dev/null

echo "Extracting archive to: $TARGET_DIR"
unzip -o "$ARCHIVE_PATH" -d "$TARGET_DIR" >/dev/null

echo
echo "Done. Extracted directory should include:"
echo "  $TARGET_DIR/datasets/Omni3D"
echo
echo "Next step in this repo:"
echo "  python setup_omni3d_data.py --source-annotations $TARGET_DIR/datasets/Omni3D --annotations-only --mode link --force"

