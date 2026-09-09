
from google import genai
import time
import cv2
import json
from PIL import Image
import os
from dotenv import load_dotenv
import anthropic
import base64

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
claude_client = anthropic.Client(api_key=CLAUDE_API_KEY)

video_filename = "March-25/case_conference_march_25_2.mp4"

def wait_for_active(client, file_name):
    while True:
        f = client.files.get(name=file_name)
        if f.state == "ACTIVE":
            return f
        if f.state == "FAILED":
            raise RuntimeError("File processing failed")
        time.sleep(2)


uploaded = client.files.upload(file=video_filename)
file_obj = wait_for_active(client, uploaded.name)

transcription = client.models.generate_content(
    model="gemini-3.5-flash", 
          contents=[file_obj,"""
          Transcribe the audio in the video into text. 
          Identify only a distinct CT image frame that shows 2 CT images and explains the diagnosis. 
          Return the response in a clean valid json format with keys 'timestamp' and 'text'. The timestamp should be in MM:SS format.
          """
]
)
transcript_text = transcription.text.strip()

if transcript_text.startswith("```json"):
    transcript_text = transcript_text.removeprefix("```json")

if transcript_text.startswith("```"):
    transcript_text = transcript_text.removeprefix("```")

if transcript_text.endswith("```"):
    transcript_text = transcript_text.removesuffix("```")

transcript_text = transcript_text.strip()

transcript_data = json.loads(transcript_text)


# print(transcript_data['text'])
# print(transcript_data['timestamp'])
def generate_mcqs():

    prompt = f"""
    Based on the following content, generate 1 multiple-choice question.
    The question should have 4 options and one  correct answer.
    The question should indicate the reason of the CT scan and ask for the most likely or least likely diagnosis based on the the explanation of the case. Don't include any more details about the patient or clinical presentation in the question.
    The options shouble be different potential diagnosis discussed in the content. 
    Format the output as a JSON list of objects, each containing:
    "question", "options" (as a list), and "answer".
    
    Content: {transcript_data['text']}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text




def extract_frame_at_time(video_path, timestamp_str, output_filename):
    """
    Extracts a single frame from a video at a specific MM:SS timestamp.
    """
    # Parse MM:SS into total seconds
    minutes, seconds = map(int, timestamp_str.split(':'))
    time_in_seconds = (minutes * 60) + seconds
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get frames per second (fps)
    #fps = cap.get(cv2.getPerspectiveTransform) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame number to jump to
    frame_id = int(time_in_seconds * fps)
    
    # Set the video position to the calculated frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    # Read the frame
    success, frame = cap.read()
    
    if success:
        # Save the frame as an image file
        cv2.imwrite(output_filename, frame)
        print(f"Successfully saved: {output_filename} (extracted from {timestamp_str})")
    else:
        print(f"Error: Could not read frame at {timestamp_str}")
        
    # Release the video object
    cap.release()


# prompt = """
# Analyze this video and identify only distinct CT image frames that show 2 CT images.
# For each unique CT view , provide:
# 1. The exact timestamp. 
# 2. A brief description of what the image shows.
# 3. Save the filename in the response as PNG.
# 4. the start time and the end time individually.
# 5. indicate if the slide reveal the best diagnosis or not with true and false and call it is_best_diagnosis. the slide that reveals the best diagnosis should contain CT images.
# Return the final response in a clean valid json format.
# """
# response = client.models.generate_content(
#     model="gemini-3.5-flash", contents=[file_obj, prompt]
# )

# text = response.text.strip()

# if text.startswith("```json"):
#     text = text.removeprefix("```json")

# if text.startswith("```"):
#     text = text.removeprefix("```")

# if text.endswith("```"):
#     text = text.removesuffix("```")

# text = text.strip()

# data = json.loads(text)

filename = "extracted_frame.png"

# timestamp = data[0]["timestamp"]
# filename = data[0]["filename"]
# extract_frame_at_time(video_filename, timestamp, filename)
extract_frame_at_time(video_filename, transcript_data['timestamp'], filename)

# for item in data:
#     timestamp = item["timestamp"]
#     filename = item["filename"]
#     is_best_diagnosis = item["is_best_diagnosis"]
#     # print(f"{timestamp}: , {is_best_diagnosis}")
#     # index = data.index(item)
#     if is_best_diagnosis:
#          extract_frame_at_time(video_filename, timestamp, filename)
#          break
   


mcqs = generate_mcqs()
print(mcqs)

with open(filename, "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = claude_client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "This slide contains two CT images. Provide the exact coordinates(x_min, y_min, x_max, y_max) for each CT image. Each CT images is surrounded by a black background. Include the black background in the coordinates only. Don't include any other dark bluish background or text outside of black background in the coordinates. y_max is within the image bounds where the black background ends. Crop out any title bar and watermark area and blue background. Return the response in a clean valid json format with names of the ct images and coordinates. The json top level key should be 'ct_images' and the value should be a list of dictionaries with keys 'name' and 'coordinates'. The coordinates should be in the format [x_min, y_min, x_max, y_max]. The names of the ct images should be 'ct_image_left' and 'ct_image_right'.",
                },
            ],
        }
    ],
)


#cropped_image = image.crop((x_min, y_min, x_max, y_max))
#print(message.content[0].text)
text = message.content[0].text.strip()

if text.startswith("```json"):
    text = text.removeprefix("```json")

if text.startswith("```"):
    text = text.removeprefix("```")

if text.endswith("```"):
    text = text.removesuffix("```")

text = text.strip()

data = json.loads(text)
# print(data)
image = Image.open(filename)

cropped_image1 = image.crop((data['ct_images'][0]["coordinates"][0], data['ct_images'][0]["coordinates"][1], data['ct_images'][0]["coordinates"][2], data['ct_images'][0]["coordinates"][3]))

# 4. Save the result
cropped_image1.save(data['ct_images'][0]["name"] + '.jpg')

cropped_image2 = image.crop((data['ct_images'][1]["coordinates"][0], data['ct_images'][1]["coordinates"][1], data['ct_images'][1]["coordinates"][2], data['ct_images'][1]["coordinates"][3]))

# 4. Save the result
cropped_image2.save(data['ct_images'][1]["name"] + '.jpg')
