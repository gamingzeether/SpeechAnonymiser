import whisper
import os
import shutil
from pathlib import Path
from pydub import AudioSegment

model = whisper.load_model("medium.en")

rawAudioDir = os.environ.get('RAW_AUDIO_DIR')
processedDir = os.environ.get('PROCESSED_DIR')
dataClientId = os.environ.get('DATA_CLIENT_ID')

for audioFile in os.listdir(rawAudioDir):
    print("Generating transcript for " + audioFile + "...")
    audioFile = "/" + audioFile
    path = Path(audioFile)
    inputFile = rawAudioDir + audioFile
    outPath = processedDir + "/whisper/" + dataClientId + "/" + path.stem
    outputMp3 = processedDir + "/clips/" + path.stem + ".mp3" 

    result = model.transcribe(inputFile, fp16=False)
    # Get transcript
    f = open(outPath + ".txt", "w")
    f.write(result["text"])
    f.close()
    # Convert to mp3 for speech anonymiser
    AudioSegment.from_file(inputFile).export(outputMp3 , format="mp3")
# Move to processed
for audioFile in os.listdir(rawAudioDir):
    audioFile = "/" + audioFile
    path = Path(audioFile)
    inputFile = rawAudioDir + audioFile
    outPath = processedDir + "/whisper/" + dataClientId + "/" + path.stem
    outputMp3 = processedDir + "/clips/" + path.stem + ".mp3"

    # Move audio file so it isnt processed again
    shutil.move(inputFile, outPath + path.suffix)
