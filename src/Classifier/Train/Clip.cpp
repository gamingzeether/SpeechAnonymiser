#include "Clip.hpp"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

#include <cstring> // Includes memcpy
#include <iostream>
#include <filesystem>
#include <dr_mp3.h>
#include <dr_wav.h>
#include <samplerate.h>

void Clip::load(int targetSampleRate) {
  size_t clipSamples, clipSampleRate;
  float* floatBuffer = NULL;

  std::string clipFullPath = getFilePath();
  if (clipFullPath == "")
    return;

  switch (type) {
  case COMMON_VOICE:
    floatBuffer = loadMP3(clipSamples, clipSampleRate, clipFullPath);
    break;
  case TIMIT:
    floatBuffer = loadWAV(clipSamples, clipSampleRate, clipFullPath);
    break;
  }
  if (floatBuffer == NULL) {
    printf("Failed to open file: %s\n", clipFullPath.c_str());
    return;
  }

  // Convert sample rate
  if (clipSampleRate == targetSampleRate) {
    for (size_t i = 0; i < clipSamples; i++) {
      buffer[i] = floatBuffer[i];
    }
    size = clipSamples;
    sampleRate = clipSampleRate;
  } else if (clipSampleRate % targetSampleRate == 0 && clipSampleRate > targetSampleRate) {
    // Integer down sample rate conversion
    int factor = clipSampleRate / targetSampleRate;
    size = clipSamples / factor;
    for (size_t i = 0; i < size; i++) {
      buffer[i] = floatBuffer[i * factor];
    }
    sampleRate = targetSampleRate;
  } else {
    SRC_DATA upsampleData = SRC_DATA();
    upsampleData.data_in = floatBuffer;
    upsampleData.input_frames = clipSamples;
    double ratio = (double)targetSampleRate / clipSampleRate;
    long outSize = (long)(ratio * clipSamples);
    upsampleData.src_ratio = ratio;
    upsampleData.data_out = buffer.data();
    upsampleData.output_frames = outSize;
    int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, 1);
    if (error) {
      std::cout << "Error while resampling: " << src_strerror(error) << '\n';
    }
    sampleRate = targetSampleRate;
    size = outSize;
  }
  free(floatBuffer);

  // Normalize volume
  float max = 0;
  for (size_t i = 0; i < size; i++) {
    const float& val = buffer[i];
    if (val > max) {
      max = val;
    }
  }
  float random = (float)rand() / RAND_MAX;
  float targetMax = 1.0f - 0.2 * (random);
  float factor = targetMax / max;
  for (size_t i = 0; i < size; i++) {
    buffer[i] *= factor;
  }

  loaded = true;
}

float* Clip::loadMP3(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path) {
  drmp3_config cfg;
  drmp3_uint64 scount;
  float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32(path.c_str(), &cfg, &scount, NULL);

  if (cfg.channels <= 0) {
    printf("%s has invalid channel count (%d)\n", path.c_str(), cfg.channels);
    return NULL;
  }
  if (cfg.sampleRate <= 0) {
    printf("%s has invalid sample rate (%d)\n", path.c_str(), cfg.sampleRate);
    return NULL;
  }

  samples = scount;
  convertMono(floatBuffer, samples, cfg.channels);
  sampleRate = cfg.sampleRate;
  return floatBuffer;
}

float* Clip::loadWAV(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path) {
  unsigned int channels, sr;
  drwav_uint64 clipSamples;
  float* floatBuffer = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &sr, &clipSamples, NULL);

  if (channels <= 0) {
    printf("%s has invalid channel count (%d)\n", path.c_str(), channels);
    return NULL;
  }
  if (sr <= 0) {
    printf("%s has invalid sample rate (%d)\n", path.c_str(), sr);
    return NULL;
  }

  samples = clipSamples;
  convertMono(floatBuffer, samples, channels);
  sampleRate = sr;
  return floatBuffer;
}

std::string Clip::getFilePath() {
  std::string path;
  switch (type) {
  case COMMON_VOICE:
    path = clipPath + tsvElements.PATH;
    break;
  case TIMIT:
    path = clipPath;
    break;
  default:
    throw("Invalid type");
    break;
  }
  if (!std::filesystem::exists(path)) {
    printf("%s does not exist\n", path.c_str());
    return "";
  }
  return path;
}

void Clip::convertMono(float* buffer, OUT size_t& length, int channels) {
  if (channels <= 1)
    return;
  
  length /= channels;
  for (size_t i = 0; i < length; i++) {
    float sum = 0;
    size_t off = i * channels;
    for (int j = 0; j < channels; j++) {
      sum += buffer[off + j];
    }
    buffer[i] = sum / channels;
  }
}
