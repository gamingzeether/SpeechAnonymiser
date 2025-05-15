# Introduction

This project is intended to achieve a middle ground between DSP based and STT-TTS based voice changers

DSP based systems are the types of voice changers that most people would think of. They use digital signal processing techniques to apply filters to speech, distorting the voice and making it harder to recognize. Although these kinds of systems can operate on low power devices and change the way a voice sounds, they don't hide other recognizable aspects of a speaker such as an accent.

STT-TTS systems use speech to text then pass the output to a text to speech synthesizer. This method removes a lot more recognizable features but brings with it some downsides.
1. The STT has to finish transcribing the text before the TTS can start speaking. Many STT solutions also have to wait for the speaker to stop before they can start transcribing. When trying to keep other people updated in an environment where things are always changing, such as in an FPS game, this latency may be unacceptable.
1. STT may struggle to accurately transcribe something if it is spoken with an accent or if a word isn't in its vocabulary.
1. Someone might want to preserve some speech information but there is no way to retrieve them from STT.
1. Depending on the quality desired, the TTS and STT especially can be very computationally expensive.

This project aims to achieve a middle ground by using an architecture similar to STT-TTS but with more specialized STT (referred to as the classifier) and TTS (referred to as the speech engine) components. By recognizing features within a small time window (frame) instead of the whole utterance, transcription can happen with less latency and computational power. Since there isn't an existing solution that can do this, this classifier is a custom solution and can be tailored to the specific needs of this project. This includes extracting extra features we may want to pass to the speech engine like pitch and stress. Additionally, as a byproduct of working with frames, there is no vocabulary that the classifier or speech engine is constrained to and can theoretically work for any language without any additional configuration.

## Releases

https://github.com/gamingzeether/SpeechAnonymiser/releases

## Process overview

1. Convert speech to frames.
1. Identify features in each frame.
1. Additional processing on features.
    - ex: changing timing, changing from one accent to another, replacing words with synonyms, transcribing for subtitles.
1. Features to speech with a speech synthesiser.

## Features

- [ ] Speech to phoneme
- [x] <img src="https://media1.tenor.com/m/-QWKmyICTLcAAAAd/cuh-guh.gif" height="100">
- [ ] Accent replacement
- [ ] Adjustable tempo
- [ ] Change word stress
- [x] Phoneme to speech

## Usage

To use, simply run the executable from a command line. `[Path to exectuable]`

Environment variables can be used to pass extra information about the system.
- `OMP_NUM_THREADS=n`: Tell omp to use n threads.
- `OMP_DYNAMIC=true`: Tell omp to adjust the number of threads automatically.
- `NONINTERACTIVE=true`: Skip prompts and use the default

The behavior of the program can be changed by appending arguments after the exectuable. Use `-h` or `--help` to display more information.

## Building

### Dependencies
- A compiler that supports c++ 17
- CMake >= 3.16.0
- vcpkg

Libraries that this project uses directly are installed by vcpkg but those dependencies may require other libraries that aren't installed by vcpkg.
Check the error logs of vcpkg if it doesn't configure.

After installing the dependencies, run

```
git clone https://github.com/gamingzeether/SpeechAnonymiser
cd SpeechAnonymiser
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release {OPTIONS}
cmake --build .
```
`{OPTIONS}` can be replaced with flags to change functionality
- `-DASAN={ON/OFF}` Compile with Address Sanitizer. (Default OFF)
- `-DOMP={ON/OFF}` Compile with OpenMP. (No effect on MSVC) (Default OFF)
- `-DAUDIO={ON/OFF}` Compile with support for audio in/output devices. (Default ON)
- `-DMKL={ON/OFF}` Compile with Intel MKL. (Currently does nothing) (Default OFF)
- `-DLTO={ON/OFF}` Compile with link time optimization. Increases compile time. (Default ON)
- `-DGUI={ON/OFF}` Compile with support for GUIs. (Default ON)
- `-DGPU={ON/OFF}` Run the classifier on GPU. Currently not implemented in mlpack. (Default OFF)

## Training

Train with `[Path to executable] -t [path to dataset]`

### Preparing datasets

#### Common Voice
(Note: some clips are too noisy and the aligner provides bad or even counterproductive alignments)
1. Download Mozilla Common Voice dataset
1. Install Montreal Forced Aligner
1. Extract dataset
1. Preprocess with `[Path to executable] -p [path to dataset] -w [work directory] -d [MFA dictionary path] -a [MFA acoustic model path] -o [transcript output]`
1. Move the transcipt folder to same folder as clips ex:
    - [Dataset]
        - clips
        - transcript
        - train.tsv
        - ...

#### TIMIT:
1. Convert WAV files from NIST to RIFF
    - Rename new files to *\_.wav ex: SA1.WAV -> SA1\_.wav
    - Note the case and underscore
    - I used `find . -name '*.WAV' | parallel -P20 sox {} '{.}_.wav'`

### Docker

To simplify preparing datasets and training, there is the option to use the docker container. It greatly simplifies the data processing and training steps. Check out the [readme](docker/README.md) for instructions on how to use it.

## More

Here are some similar projects that I found while working on this: 
- [Speaker de-identification using diphone recognition and speech synthesis](https://lmi.fe.uni-lj.si/wp-content/uploads/2023/05/Speakerde-identificationusingdiphonerecognitionandspeechsynthesis.pdf)
- [speaker-anonymization](https://github.com/digitalphonetics/speaker-anonymization)
