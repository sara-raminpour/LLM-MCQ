import subprocess
import os
import shutil
from PIL import Image, ImageChops, ImageStat
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
import base64
import json
import anthropic

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
claude_client = anthropic.Client(api_key=CLAUDE_API_KEY)

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

endpoint = AZURE_OPENAI_ENDPOINT
model_name = "gpt-6-astra"
api_key = AZURE_OPENAI_KEY
api_version = "2024-12-01-preview"


client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

def v2s(video_path):
    tempdir = Path("extracted_frames")
    tempdir.mkdir(exist_ok=True)
    frames_folder = tempdir / "frames"
    extract_slides(video_path, frames_folder)
    

def extract_slides(video_path, frames_folder):
    """ Fill a folder with video keyframes """
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder)
    script = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
       "-fps_mode", "passthrough",
        "-vf", r"select=eq(pict_type\,PICT_TYPE_I)",
        "-f", "image2",
        os.path.join(frames_folder, "%03d.jpeg")
    ]
    print("Extracting frames...")
    subprocess.run(script, capture_output=True, text=True)
    remove_duplicates(frames_folder)
    remove_nonct(frames_folder)
    extract_images_from_frame(frames_folder)

def remove_duplicates(frames_folder):
    """ Goes through each file in the passed folder and removes if difference ratio is less than 2.0"""
    print("Removing duplicates...")
    filename_list = []
    for filename in os.listdir(frames_folder):
        filename_list.append(filename)
    filename_list.sort()
    last_frame = Image.open(frames_folder / filename_list[0])
    removal_list = []
    for i in range(1, len(filename_list)):
        cur_frame = Image.open(frames_folder / filename_list[i])
        diff_img = ImageChops.difference(cur_frame, last_frame)
        stat = ImageStat.Stat(diff_img)
        diff_percent = 100 * (sum(stat.mean) / (len(stat.mean) * 255))
        if diff_percent <= 2.0:
            removal_list.append(frames_folder / filename_list[i - 1])
        last_frame = cur_frame
    for image in removal_list:
        os.remove(image)

def remove_nonct(frames_folder):
    """Goes through each image and removes slides without 2 CT images."""
    print("Removing non-CT slides...")
    filename_list = sorted(os.listdir(frames_folder))

    remove_list = []

    for filename in filename_list:
        image_path = frames_folder / filename

        print(f"Analyzing {filename}...")

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image file and determine if it contains 2 CT images. Return true if it does, false otherwise."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )

        result = response.choices[0].message.content.strip()
        print(f"  {filename}: {result}")
        try:
            if result != "true":
                remove_list.append(image_path)
        except ValueError:
            print(f"  Could not interpret response for {filename}")

    # Remove slides that don't contain exactly 2 CT images
    for image_path in remove_list:
        print(f"Removing: {image_path.name}")
        os.remove(image_path)

def extract_images_from_frame(frames_folder):

    filename_list = sorted(os.listdir(frames_folder))
    os.makedirs(frames_folder / "ct", exist_ok=True)
    ct_path = frames_folder / "ct"
    for filename in filename_list:
            image_path = frames_folder / filename
    
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

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
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""This slide contains two CT images."
                                        "Provide the exact coordinates(x_min, y_min, x_max, y_max) for each CT image. "
                                        "Each CT images is surrounded by a black background. Include the black background in the coordinates only. "
                                        "Don't include any other dark bluish background or text outside of black background in the coordinates. "
                                        "y_max is within the image bounds where the black background ends. "
                                        "Crop out any title bar and watermark area and blue background. "
                                        "Return the response in a clean valid json format with names of the ct images and coordinates. "
                                        "The json top level key should be {filename} and the value should be a list of dictionaries with keys 'name' and 'coordinates'. "
                                        "The coordinates should be in the format [x_min, y_min, x_max, y_max]. "
                                        "The names of the ct images should be {Path(filename).stem}+'_left' and {Path(filename).stem}+'_right'."""
                            },
                        ],
                    }
                ],
            )

            text = message.content[0].text.strip()

            if text.startswith("```json"):
                text = text.removeprefix("```json")

            if text.startswith("```"):
                text = text.removeprefix("```")

            if text.endswith("```"):
                text = text.removesuffix("```")

            text = text.strip()

            data = json.loads(text)

            image = Image.open(image_path)
            

            print(data)

            cropped_image1 = image.crop((data[filename][0]["coordinates"][0], data[filename][0]["coordinates"][1], data[filename][0]["coordinates"][2], data[filename][0]["coordinates"][3]))
            cropped_image1.save(ct_path / f"{data[filename][0]['name']}.jpg")

            cropped_image2 = image.crop((data[filename][1]["coordinates"][0], data[filename][1]["coordinates"][1], data[filename][1]["coordinates"][2], data[filename][1]["coordinates"][3]))
            cropped_image2.save(ct_path / f"{data[filename][1]['name']}.jpg")
    choose_best_ct_azure(diagnosis, ct_path)

def choose_best_ct_azure(dx,path):
    """Choose the best CT image based on diagnostic information."""
    ct_folder = Path(path)
    filename_list = sorted(os.listdir(ct_folder))

    content = [ 
        { 
            "type": "text", 
            "text": ( 
                f"""You are an AI assistant. Look at the attached images (filenames: {filename_list}).
                Choose the two most complementary CT images for teaching this diagnosis. 
                The clinical diagnosis for this case is {dx}. 
                Consider diagnostic findings, clarity, anatomy, and whether the images show different useful features. 
                Avoid duplicate or near-duplicate images. Do not infer findings that are not visible.
                Return the exact filename of two images."""
            ), 
        } 
    ]
    for filename in filename_list:
        image_path = ct_folder / filename
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            }                      
        })
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )
    print("Choosing best CT images using Azure OpenAI...")
    print(response.choices[0].message.content)


    
video_path = "March-25/case_conference_march_25_1.mp4"
diagnosis = "Paraganglioma"  
v2s(video_path)    
