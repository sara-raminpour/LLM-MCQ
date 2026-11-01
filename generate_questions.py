
from google import genai
import time
import json
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def wait_for_active(client, file_name):
    while True:
        f = client.files.get(name=file_name)
        if f.state == "ACTIVE":
            return f
        if f.state == "FAILED":
            raise RuntimeError("File processing failed")
        time.sleep(2)

def generate_transcript(video_filename):
    uploaded = client.files.upload(file=video_filename)
    file_obj = wait_for_active(client, uploaded.name)

    transcription = client.models.generate_content(
        model="gemini-3.5-flash", 
            contents=[file_obj,"Transcribe the audio in the video into text."])

    return transcription.text.strip()



def generate_mcqs(transcript_text):

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


def main(video_filename: str):
    
    transcript_text = generate_transcript(video_filename)
    mcqs = generate_mcqs(transcript_text)
    print(mcqs)
    text = mcqs.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json")

    if text.startswith("```"):
        text = text.removeprefix("```")

    if text.endswith("```"):
        text = text.removesuffix("```")

    text = text.strip()

    data = json.loads(text)

    dx = data[0]['answer']
    return dx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generating MCQ from video transcripts.")
    parser.add_argument('video_filename', type=str, help='Path to the video file')
    
    args = parser.parse_args()
    main(args.video_filename)
