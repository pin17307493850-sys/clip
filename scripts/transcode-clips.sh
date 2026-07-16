#!/bin/sh
set -eu

dir="$1"
count=0
for file in "$dir"/*.mp4; do
    [ -f "$file" ] || continue
    codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$file")
    if [ "$codec" = "h264" ]; then
        echo "skipped (already h264): $(basename "$file")"
        continue
    fi
    temp="${file%.mp4}.h264.tmp.mp4"
    rm -f "$temp"
    ffmpeg -v error -i "$file" -map 0:v:0 -map 0:a? \
        -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
        -c:a aac -b:a 128k -movflags +faststart -y "$temp"
    mv "$temp" "$file"
    count=$((count + 1))
    echo "converted $count: $(basename "$file")"
done
echo "TOTAL=$count"
