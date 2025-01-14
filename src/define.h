#pragma once
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS

#define WIDTH 280
#define HEIGHT 1010
#define POS_X 1640
#define POS_Y 30
#define VERTICAL
//#define HIDE_CONSOLE

typedef float INPUT_TYPE;
#define INPUT_FORMAT RTAUDIO_FLOAT32
#define INPUT_SCALE  1.0

typedef float OUTPUT_TYPE;
#define OUTPUT_FORMAT RTAUDIO_FLOAT32
#define OUTPUT_SCALE  1.0

#define INPUT_BUFFER_TIME 0.2
#define INPUT_BUFFER_SIZE 128
#define OUTPUT_BUFFER_SIZE 1024

#define FFT_FRAME_SAMPLES 512
#define FFT_FRAME_SPACING 160
#define CONTEXT_FORWARD 7
#define CONTEXT_BACKWARD 13
#define FFT_FRAMES (CONTEXT_FORWARD + 1 + CONTEXT_BACKWARD)
#define FFT_REAL_SAMPLES (FFT_FRAME_SAMPLES / 2)
#define MEL_BINS 64
#define FRAME_SIZE 24
#define VEC4_COUNT (FRAME_SIZE / 4)
#define MMAX_EXAMPLE_F 3
#define ACTIVITY_WIDTH 5
#define INFERENCE_FRAMES 2 // Classify once every x frames
#define DELTA_DISTANCE 4 // Get delta from x frames back; must be less than FFT_FRAMES
#define AVG_FRAMES 4

#define CLIP_LENGTH 15
