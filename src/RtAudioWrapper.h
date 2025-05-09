#pragma once

#ifdef AUDIO

#include <rtaudio/RtAudio.h>

#else

// Stub to include instead of RtAudio if audio support is not required or unusable

#include <string>

#define RTAUDIO_FLOAT32 0

// Empty structs to prevent errors
class RtAudio {
public:
  struct StreamParameters {
    int nChannels;
  };
  struct StreamOptions {};

  const std::string getErrorText(...) {return "";};
  int startStream(...) {return 0;};
  int openStream(...) {return 0;};
};
struct RtAudioStreamStatus {};

#endif
