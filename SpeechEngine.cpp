#include "SpeechEngine.h"

#include <type_traits>
#include <assert.h>
#include <algorithm>
#include <numbers>
#include <iostream>
#include <thread>
#include <string>
#include "ClassifierHelper.h"

#define ENGINE_STEP_FRAMES 256
#define ENGINE_TIMESTEP ((double)ENGINE_STEP_FRAMES / sampleRate)
#define PHASE_STEP_AMOUNT(t) (-4.0 * std::numbers::pi * t)
#define FADE_SAMPLES 256

void SpeechEngine::pushFrame(const SpeechFrame& frame) {
	int frameAnim = phonToAnim[frame.phoneme];
	int currentAnim = articAnim.currentAnim();
	if (frameAnim != currentAnim) {
		articAnim.startAnimation(frameAnim);
	}
}

void SpeechEngine::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {
	assert(nFrames > 0);
	double bufferDuration = (double)nFrames / sampleRate;
	int offset = 0;
	double phaseOffset = 0;
	double gain = 0.02;
	for (int t = 0; t < nFrames; t += ENGINE_STEP_FRAMES) {
		articAnim.stepTime(ENGINE_TIMESTEP);
		for (int i = 0; i < std::extent<decltype(articulators)>::value; i++) {
			SpeechArticulator& artic = articulators[i];
			int index = artic.animIndex;
			if (index >= 0) {
				float target = articAnim.getAnimationPos(index);
				artic.setTarget(target);
			}
			artic.step(ENGINE_TIMESTEP);
		}

		// Get fft input from position of articulators
		{
			float flowRate = pressure().getPosition();
			float sysPressure = systemPressure().getPosition();
			float frequency = baseFrequency().getPosition();
			float tongue1 = tongueBase().getPosition();
			float tongue2 = tongueMid().getPosition();
			float tongue3 = tongueTip().getPosition();
			float jawPos = jaw().getPosition();
			float lipPos = lip().getPosition();
			// Move physically animated articulators
			float effectiveFlowRate;
			{
				float flowOpening = std::fminf(tongue1, std::fminf(tongue2, std::fminf(tongue3, std::fminf(jawPos + lipPos, 0.1))));
				flowOpening = std::fminf(1.0f, flowOpening);
				flowRate = (flowOpening > 0.2) ? 0.0 : flowRate;
				systemPressure().setTarget(flowRate);
				float pressureDelta = sysPressure - prevPressure;
				prevPressure = sysPressure;
				effectiveFlowRate = std::fmaxf(flowRate, pressureDelta);
			}
			int baseFreqBin = freqToBin(frequency);
			baseFreqBin = std::min(std::max(1, baseFreqBin), fftBins - 1);
			assert(baseFreqBin < fftBins);

			amplitude[baseFreqBin] = effectiveFlowRate * gain;
		}
		
		// Step fft phase
		phaseOffset += PHASE_STEP_AMOUNT(ENGINE_TIMESTEP);
		// Write amplitude and phase as complex numbers
		{
			for (int i = 0; i < fftBins; i++) {
				if (amplitude[i] > 0.00001) {
					double binPhase = (phase + phaseOffset) * i;
					fftwIn[i][0] = amplitude[i] * std::sin(binPhase);
					fftwIn[i][1] = amplitude[i] * std::cos(binPhase);
				} else {
					fftwIn[i][0] = 0;
					fftwIn[i][1] = 0;
				}
			}
		}

		// Reset amplitude
		for (int i = 0; i < fftBins; i++) {
			amplitude[i] *= 0.5;
		}

		// Might not be ideal since this is using only 256 frames per iteration
		// Use sum of sines instead?
		// Compute inverse fft
		fftwf_execute(fftwPlan);

		// Copy audio to buffer
		{
			for (int i = 0; i < ENGINE_STEP_FRAMES; i++) {
				float value = fftwOut[i];

				if (i <= FADE_SAMPLES) {
					float fade = (float)i / FADE_SAMPLES;
					value = std::lerp(fadeSamples[i], value, fade);
				}

				for (int c = 0; c < channels; c++) {
					*outputBuffer++ = value;
				}
			}
			memcpy(fadeSamples, fftwOut + ENGINE_STEP_FRAMES, FADE_SAMPLES * sizeof(float));
			offset += ENGINE_STEP_FRAMES;
		}
	}
	phase = std::fmod(phase + PHASE_STEP_AMOUNT(bufferDuration), 2 * std::numbers::pi);
}

float SpeechEngine::_random() {
	std::uniform_real_distribution<float> randomDist = std::uniform_real_distribution<float>(-1, 1);
	return randomDist(randomEngine);
}

SpeechEngine::SpeechEngine() {
	sampleRate = 16000;
	channels = 1;
	_init();
}

SpeechEngine::SpeechEngine(int sr, int ch) {
	assert(sr > 0);
	assert(ch > 0);
	sampleRate = sr;
	channels = ch;
	_init();
}

SpeechEngine::SpeechArticulator::SpeechArticulator(float proportional, float integra, float derivative, float rand, int index) {
	_init(proportional, integra, derivative, rand, index);
}

SpeechEngine::SpeechArticulator::SpeechArticulator(std::string path, int index) {
	JSONHelper json;
	// Open or initalize with default
	{
		bool jsonOpen = json.open(path.c_str(), _jsonVersion);
		if (!jsonOpen) {
			json["proportional"] = 1.0;
			json["integral"] = 1.0;
			json["derivative"] = 1.0;
			json["random"] = 1.0;
			json.save();
		}
	}

	// Load values from json
	float proportional, integral, derivative, random;
	proportional = json["proportional"].get_real();
	integral = json["integral"].get_real();
	derivative = json["derivative"].get_real();
	random = json["random"].get_real();
	_init(proportional, integral, derivative, random, index);
}

void SpeechEngine::SpeechArticulator::step(float dTime) {
	float error = target - position;
	error += _random() * random;
	integral += error * dTime;
	float derivative = (error - prevError) / dTime;
	prevError = error;

	float pid = (error * p) + (integral * i) + (derivative * d);

	velocity += pid * dTime;
	position += velocity * dTime;
}

void SpeechEngine::SpeechArticulator::_init(float proportional, float integra, float derivative, float rand, int index) {
	target = 0;
	position = 0;
	velocity = 0;
	prevError = 0;
	integral = 0;
	random = rand;

	p = proportional;
	i = integral;
	d = derivative;

	animIndex = index;
}

void SpeechEngine::_init() {
#pragma region Logger
	logger = Logger();
	logger.addStream(Logger::Stream("speech_engine.log").
		outputTo(Logger::VERBOSE).
		outputTo(Logger::INFO).
		outputTo(Logger::WARNING).
		outputTo(Logger::ERR).
		outputTo(Logger::FATAL));
	logger.addStream(Logger::Stream(std::cout).
		outputTo(Logger::INFO).
		outputTo(Logger::WARNING).
		outputTo(Logger::ERR).
		outputTo(Logger::FATAL));

	logger.log("Starting", Logger::VERBOSE);
	logger.log(std::format("Sample rate: {}", sampleRate), Logger::VERBOSE);
	logger.log(std::format("Channels: {}", channels), Logger::VERBOSE);
#pragma endregion

#pragma region Animations
	_initArticulators();
	auto& ips = ClassifierHelper::instance().inversePhonemeSet;
	for (int i = 0; i < ips.size(); i++) {
		const std::string& name = ips[i];
		articAnim.loadGroup(std::format("animations/phonemes/{}_anim.json", name), name);
	}
	articAnim.finalize();
	for (int i = 0; i < ips.size(); i++) {
		const std::string& name = ips[i];
		phonToAnim[i] = articAnim.getAnimation(name);
	}
	// Disable animations
	systemPressure().animIndex = -1;

	// Set default anim to silence
	size_t initPhoneme = ClassifierHelper::instance().phonemeSet[ClassifierHelper::instance().customHasher(L"")];
	articAnim.startAnimation(initPhoneme);
#pragma endregion

#pragma region FFT
	fftBins = sampleRate / 2;
	fftwIn = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * fftBins);
	fftwOut = (float*)fftw_malloc(sizeof(float) * fftBins * 2);
	fftwPlan = fftwf_plan_dft_c2r_1d(fftBins, fftwIn, fftwOut, FFTW_MEASURE);

	for (int i = 0; i < fftBins; i++) {
		fftwf_complex& cplx = fftwIn[i];
		cplx[0] = 0;
		cplx[1] = 0;
	}

	amplitude = new float[fftBins];
	for (int i = 0; i < fftBins; i++) {
		amplitude[i] = 0;
	}
#pragma endregion

	fadeSamples = new float[FADE_SAMPLES];
	
	logger.log("Initalized", Logger::VERBOSE);
}

void SpeechEngine::_initArticulators() {
	pressure() = SpeechArticulator("articulators/pressure.json", Animator::PRESSURE);
	systemPressure() = SpeechArticulator("articulators/system_pressure.json", Animator::SYSTEM_PRESSURE);
	baseFrequency() = SpeechArticulator("articulators/base_frequency.json", Animator::BASE_FREQUENCY);
	tongueBase() = SpeechArticulator("articulators/tongue_base.json", Animator::TONGUE_BASE);
	tongueMid() = SpeechArticulator("articulators/tongue_mid.json", Animator::TONGUE_MID);
	tongueTip() = SpeechArticulator("articulators/tongue_tip.json", Animator::TONGUE_TIP);
	jaw() = SpeechArticulator("articulators/jaw.json", Animator::JAW);
	lip() = SpeechArticulator("articulators/lip.json", Animator::LIP);
}
