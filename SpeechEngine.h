#pragma once

#include "common_inc.h"

#include <chrono>
#include <random>
#include <unordered_map>
#include <fftw3.h>
#include "structs.h"
#include "define.h"
#include "Logger.h"
#include "JSONHelper.h"
#include "Animator.h"

class SpeechEngine {
public:
	void pushFrame(const SpeechFrame& frame);
	void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames);
	static float _random();

	SpeechEngine();
	SpeechEngine(int sr, int ch);
private:
	class SpeechArticulator {
	public:
		int animIndex = -1;
		
		void setTarget(float t) { target = t; };
		float getPosition() { return position; };
		void step(float dTime);

		SpeechArticulator(float proportional = 1, float integral = 1, float derivative = 1, float rand = 1, int index = -1);
		SpeechArticulator(std::string path, int index = -1);
	private:
		// state
		float position;
		float velocity;
		float prevError;
		float integral;

		// parameters
		float target;
		float random;

		float p;
		float i;
		float d;

		static const int _jsonVersion = 0;
		void _init(float proportional, float integral, float derivative, float rand, int index);
	};
	class TractSegment {
	public:
		float step(float input);
		void setLossFactor(float l) { loss = 1.0 - l; };
		inline float pressure(int index);
		void setRadius(int index, float rad);

		TractSegment() : TractSegment(2) {};
		TractSegment(int l) : length(l) {
			assert(l >= 2);
			forward1 = new float[l];
			backward1 = new float[l];
			forward2 = new float[l];
			backward2 = new float[l];
			radius = new float[l];
			scattering = new float[l - 1];
			loss = 1;

			for (int i = 0; i < l; i++) {
				setRadius(i, 1);
			}
		};
	private:
		const int length;
		bool currentBuffer1 = true;
		float* forward1;
		float* backward1;
		float* forward2;
		float* backward2;
		float* radius;
		float* scattering;
		float loss;

		void stepBuffer(float* buf);
	};
	class VocalTract {
	public:

	private:

	};

	Logger logger;

	int sampleRate;
	int channels;
	int fftBins;

	inline static std::default_random_engine randomEngine = std::default_random_engine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());

	std::unordered_map<size_t, int> phonToAnim;

	float* amplitude;
	double phase;
	float* fadeSamples;

	fftwf_complex* fftwIn;
	float* fftwOut;
	fftwf_plan fftwPlan;

#pragma region Articulators
	Animator articAnim;
	inline SpeechArticulator& pressure() { return articulators[0]; }; // Flow rate
	inline SpeechArticulator& systemPressure() { return articulators[1]; }; // Pressure of the system
	inline SpeechArticulator& baseFrequency() { return articulators[2]; }; // Vocal cords
	inline SpeechArticulator& tongueBase() { return articulators[3]; }; // Back of tongue
	inline SpeechArticulator& tongueMid() { return articulators[4]; }; // Middle part of tongue
	inline SpeechArticulator& tongueTip() { return articulators[5]; }; // Tongue tip
	inline SpeechArticulator& jaw() { return articulators[6]; }; // Jaw
	inline SpeechArticulator& lip() { return articulators[7]; }; // Lips
	SpeechArticulator articulators[8];
	float prevPressure = 0;
#pragma endregion

	void _init();
	void _initArticulators();
	int freqToBin(float frequency) { return (int)((frequency * 0.5f) + 0.5f); };
};
