# Speech Anonymiser

This project is intended to modify parts of spoken language to make it harder to identify the person behind it. Every person has their own speaking quirks that make them identifiable through their voice. The easiest feature to recognize is probably the way their voice sounds which is why people wanting to hide their identities may use a voice changer. The voice isn't the only feature that can help to give away their identity. Things like accents, commonly used phrases or words, how words are stressed, pacing, and loudness are unique to each person but aren't usually changed by a voice changer. 

Voice changer systems that use a STT and TTS solve many of the issues above but have their own downsides. 
- Latency: The STT has to finish transcribing the text before the TTS can start speaking. Many solutions to STT wait for the utterance to be completed before they can start transcribing. When trying to keep other people updated in an environment where things are always changing, such as in an FPS game, this latency may be unacceptable.
- Accuracy: Depending on factors like accents and words that aren't in the dictionary, the STT may not accurately transcribe the word. 
    - ex: "Haruka" -> "How to car"
- Losing too many features: Although features like pitch, tempo, and stresses may help make someone's speech identifiable, they also play an important role in conveying meaning and emotion. 
- Computing resources: Many commerical speech recognition systems use massive neural networks to transcribe audio which can use a lot of computational resources.
    - Doing it on another device may add additional latency and costs. 

This project aims to provide as much anonymity as STT-TTS voice changers while having fewer downsides.
- Latency: By windowing the speech into frames and identifying the phoneme in each frame, transcription can be done as the frames arrive.
- Accuracy: By using phonemes instead of whole words, the system does not depend on a dictionary. As a side effect, accents have less of an effect on transcription accuracy.
- Losing too many features: Removing features is mostly done through modifying the transcribed phonemes rather than losing them through the STT-TTS process. This approach allows for more control over what is changed.
- Computing resources: Because it transcribes frames to phonemes instead of the whole utterance to words, the model for transcription can be much smaller.

Also see: 
- [Speaker de-identification using diphone recognition and speech synthesis](https://lmi.fe.uni-lj.si/wp-content/uploads/2023/05/Speakerde-identificationusingdiphonerecognitionandspeechsynthesis.pdf)
  - Proposes DROPSY, a similar method that uses diphone recognition and focuses on hiding the voice
- [speaker-anonymization](https://github.com/digitalphonetics/speaker-anonymization)

# Process

1. Convert speech to frames
1. Identify phoneme in each frame
1. Additional processing on phonemes
    - ex: changing spacing between phonemes, changing from one accent to another, replacing words with synonyms, transcribing for subtitles
1. Phonemes to speech with a speech synthesiser

# Features

- [x] Speech to phoneme
- [x] <img src="https://media1.tenor.com/m/-QWKmyICTLcAAAAd/cuh-guh.gif" height="100">
- [ ] Accent replacement
- [ ] Adjustable tempo
- [ ] Change word stress
- [ ] Phoneme to speech

# Usage

Run the executable
### Flags
- `-t` or `--train` Start in training mode
- `-p` or `--preprocess` Process dataset before training
- `-h` or `--help` Shows help

No flags will start the program in the default mode

# Releases

https://github.com/gamingzeether/SpeechAnonymiser/releases


# Dependencies
- A compiler that supports c++ 20 (and std::format)
- CMake >= 3.16.0
- vcpkg

Libraries that this project uses directly are installed by vcpkg but those dependencies may require other libraries that aren't installed by vcpkg.
The below commands *should* install all of the required libraries.
### Debian/Ubuntu
```
sudo apt-get install -y \
    gfortran \
    curl \
    pkg-config \
    libxinerama-dev \
    libxcursor-dev \
    xorg-dev \
    libglu1-mesa-dev
```

For more information, check out [the Dockerfile](docker/Dockerfile)

### Windows
Should have the required libraries but I haven't tested this

# Building
Run
```
git clone https://github.com/gamingzeether/SpeechAnonymiser
cd SpeechAnonymiser
mkdir out
cmake -B out -DCMAKE_BUILD_TYPE=Release {OPTIONS}
cmake --build out
```
`{OPTIONS}` can be replaced with flags to change functionality
- `-DASAN={ON/OFF}` Compile with Address Sanitizer. (Default OFF)
- `-DOMP={ON/OFF}` Compile with OpenMP. (No effect on MSVC) (Default OFF)
- `-DAUDIO={ON/OFF}` Compile with audio in/output. (Default ON)
- `-DMKL={ON/OFF}` Compile with Intel MKL. (Currently does nothing) (Default OFF)
- `-DLTO={ON/OFF}` Compile with link time optimization. Increases compile time. (Default ON)

# Preparing datasets

If using Common Voice: (Note: some clips are too noisy and the aligner provides bad alignments)
1. Download Mozilla Common Voice dataset
1. Install Montreal Forced Aligner
1. Extract dataset
1. Preprocess with `-p [path to dataset] -w [work directory] -d [MFA dictionary path] -a [MFA acoustic model path] -o [transcript output]`
1. Move the transcipt folder to same folder as clips ex:
    - [Dataset]
        - clips
        - transcript
        - train.tsv
        - ...

If using TIMIT:
1. Convert WAV files from NIST to RIFF
    - Rename new files to *\_.wav ex: SA1.WAV -> SA1\_.wav
    - I used `find . -name '*.WAV' | parallel -P20 sox {} '{.}_.wav'`

# Training

1. Train with `-t [path to dataset]`

# Docker

To simplify preparing datasets and training, there is the option to use docker. It contains scripts that processes unlabeled recordings into a usable dataset. It can also be used to train the phoneme classifier. Check out the [readme](docker/README.md) for instructions on how to use it.

# Want to contribute? / Have questions? / Something doesn't work?

Make an issue or pull request
