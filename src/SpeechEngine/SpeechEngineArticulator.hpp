#pragma once

#include "SpeechEngine.hpp"

#include "Animator.hpp"

// Articulator based speech synthesiser
// Still work in progress

class SpeechEngineArticulator : public SpeechEngine {
public:
	SpeechEngineArticulator() {};

	virtual void pushFrame(const SpeechFrame& frame);
	virtual void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames);

	virtual SpeechEngineArticulator& configure(std::string file);
protected:
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
			forward1 = new float[length];
			backward1 = new float[length];
			forward2 = new float[length];
			backward2 = new float[length];
			radius = new float[length];
			scattering = new float[length - 1];
			loss = 1;

			for (int i = 0; i < length; i++) {
				float rand = SpeechEngine::_random();
				setRadius(i, 1 + rand * 0.1);
				forward1[i] = 0;
				backward1[i] = 0;
				forward2[i] = 0;
				backward2[i] = 0;
			}
		};
	private:
		int length;
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
		float step(float input);

		VocalTract() : VocalTract(16000) {};
		VocalTract(int ln) {
			ts1 = TractSegment(ln);
			ts1.setLossFactor(0.3 / ln);
		}
	private:
		TractSegment ts1;
	};

	virtual void _init();
	void _initArticulators();

	std::unordered_map<size_t, int> phonToAnim;

	float phase = 0;
	float* wavetable;
	VocalTract vocalTract;

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
};
