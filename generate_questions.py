
from google import genai
import time
import json
from PIL import Image
import os
from dotenv import load_dotenv
from google.genai import types


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)



video_filename = "March-25/case_conference_march_25_11.mp4"

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
          contents=[file_obj,"Transcribe the audio in the video into text."])

transcript_text = transcription.text.strip()


def generate_mcqs():

    prompt = f"""
    Based on the following content, generate 1 multiple-choice question.
    Follow the instructions carefully:
    1) The question should have 4 options and one correct answer.
    The question should indicate the possible clinical reason of the performed CT scan and ask for the most likely or least likely diagnosis based on the the explanation of the case. Don't include any more details about the patient or clinical presentation in the question or any imaging findings. 
    Don't use the phrase 'Based on the explanation of the case' in the question.
    2) The options shouble be different potential diagnosis discussed in the content. 
    3) Format the output as a JSON list of objects, each containing:
    "question", "options" (as a list), and "answer". 
    For example, 'Patient came with fever and cough, what is the most likely diagnosis?' or 'Patient came with fever and cough, what is the least likely diagnosis?'

    Content: {transcript_text}
    """


    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text



mcqs = generate_mcqs()
print(mcqs)