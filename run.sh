cd "$(dirname "$0")"
for video in Input_March_25/*.mp4; do
    python3 vid2slides.py "$video"
done
#python3 vid2slides.py input/case_conference_march_25_11.mp4