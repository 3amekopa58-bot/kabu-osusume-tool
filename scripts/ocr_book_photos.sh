#!/bin/bash
# 本の写真（HEIC）をまとめてOCRし、1つのテキストファイルにまとめるスクリプト。
# 縦書き日本語に対応した tesseract の jpn_vert モデルを使用。
#
# 使い方:
#   ./ocr_book_photos.sh <写真フォルダ> <開始番号> <終了番号> <出力テキストファイル>
# 例:
#   ./ocr_book_photos.sh "/Users/kopamon/Library/Mobile Documents/com~apple~CloudDocs/株" 3610 3621 ocr_output.txt

set -e

PHOTO_DIR="$1"
START="$2"
END="$3"
OUT_FILE="$4"
TMPDIR="${TMPDIR:-/tmp}/ocr_book_photos_$$"

mkdir -p "$TMPDIR"
> "$OUT_FILE"

for n in $(seq "$START" "$END"); do
  heic="$PHOTO_DIR/IMG_${n}.HEIC"
  if [ ! -f "$heic" ]; then
    heic="$PHOTO_DIR/IMG_${n}.heic"
  fi
  if [ ! -f "$heic" ]; then
    echo "警告: IMG_${n}.HEIC が見つかりません" >&2
    continue
  fi

  jpg="$TMPDIR/IMG_${n}.jpg"
  sips -s format jpeg -Z 2400 "$heic" --out "$jpg" >/dev/null 2>&1

  echo "===== IMG_${n} =====" >> "$OUT_FILE"
  tesseract "$jpg" - -l jpn_vert 2>/dev/null >> "$OUT_FILE"
  echo "" >> "$OUT_FILE"
  echo "OCR完了: IMG_${n}"
done

rm -rf "$TMPDIR"
echo "完了: $OUT_FILE"
