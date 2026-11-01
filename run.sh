cd "$(dirname "$0")"
for video in Input_March_25/*.mp4; do
    python3 vid2slides.py "$video"
done
