#include "SpeechEngineArticulator.h"

#include <type_traits>
#include <assert.h>
#include <algorithm>
#include <numbers>
#include <iostream>
#include <thread>
#include "ClassifierHelper.h"
#include "Global.h"

#define ENGINE_STEP_FRAMES 256
#define ENGINE_TIMESTEP ((double)ENGINE_STEP_FRAMES / sampleRate)
#define PHASE_STEP_AMOUNT(t) (-4.0 * std::numbers::pi * t)

SpeechEngineArticulator& SpeechEngineArticulator::configure(std::string file) {
	_init();
	return *this;
}

void SpeechEngineArticulator::pushFrame(const SpeechFrame& frame) {
	int frameAnim = phonToAnim[frame.phoneme];
	int currentAnim = articAnim.currentAnim();
	if (frameAnim != currentAnim) {
		articAnim.startAnimation(frameAnim);
	}
}

void SpeechEngineArticulator::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {
	assert(nFrames > 0);
	double bufferDuration = (double)nFrames / sampleRate;
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

		// Get input from position of articulators
		float glotFreq, effectiveFlowRate;
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
			{
				float flowOpening = std::fminf(tongue1, std::fminf(tongue2, std::fminf(tongue3, std::fminf(jawPos + lipPos, 0.1))));
				flowOpening = std::fminf(1.0f, flowOpening);
				flowRate = (flowOpening > 0.2) ? 0.0 : flowRate;
				systemPressure().setTarget(flowRate);
				float pressureDelta = sysPressure - prevPressure;
				prevPressure = sysPressure;
				effectiveFlowRate = std::fmaxf(flowRate, pressureDelta);
			}

			glotFreq = std::max(0.0f, frequency);
		}

		// Generate audio
		{
			auto oBufStart = outputBuffer;
			for (int i = 0; i < ENGINE_STEP_FRAMES; i++) {
				phase += glotFreq;
				float value = wavetable[(int)phase % sampleRate] * effectiveFlowRate;
				float out = vocalTract.step(value) * gain;
				//std::printf("%f\n", out);
				//if (std::abs(out) > 0.75) {
				//	std::cout << out;
				//	throw;
				//}

				for (int c = 0; c < channels; c++) {
					*outputBuffer++ = out;
				}
			}
			phase = fmod(phase, sampleRate);
		}
	}
}

#pragma region Articulator
SpeechEngineArticulator::SpeechArticulator::SpeechArticulator(float proportional, float integra, float derivative, float rand, int index) {
	_init(proportional, integra, derivative, rand, index);
}

SpeechEngineArticulator::SpeechArticulator::SpeechArticulator(std::string path, int index) {
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

void SpeechEngineArticulator::SpeechArticulator::step(float dTime) {
	float error = target - position;
	error += _random() * random;
	integral += error * dTime;
	float derivative = (error - prevError) / dTime;
	prevError = error;

	float pid = (error * p) + (integral * i) + (derivative * d);

	velocity += pid * dTime;
	position += velocity * dTime;
}

void SpeechEngineArticulator::SpeechArticulator::_init(float proportional, float integra, float derivative, float rand, int index) {
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
#pragma endregion

#pragma region Tract segments
float SpeechEngineArticulator::TractSegment::pressure(int index) {
	float* forwardRd, * backwardRd;
	if (currentBuffer1) {
		forwardRd = forward1;
		backwardRd = backward1;
	} else {
		forwardRd = forward2;
		backwardRd = backward2;
	}
	return (forwardRd[index] + backwardRd[index]) * 0.5;
};

void SpeechEngineArticulator::TractSegment::setRadius(int index, float rad) {
	radius[index] = rad;
	float rsqri = radius[index] * radius[index];

	if (index > 0) {
		float rsqrb = radius[index - 1] * radius[index - 1];
		scattering[index - 1] = (rsqrb - rsqri) / (rsqrb + rsqri);
	}
	if (index < length - 1) {
		float rsqrf = radius[index + 1] * radius[index + 1];
		scattering[index] = (rsqri - rsqrf) / (rsqri + rsqrf);
	}
}

float SpeechEngineArticulator::TractSegment::step(float input) {
	float* forwardRd, * forwardWr, * backwardRd, * backwardWr;
	if (currentBuffer1) {
		forwardRd = forward1;
		forwardWr = forward2;
		backwardRd = backward1;
		backwardWr = backward2;
	} else {
		forwardRd = forward2;
		forwardWr = forward1;
		backwardRd = backward2;
		backwardWr = backward1;
	}
	currentBuffer1 = !currentBuffer1;

	float forwardBack = forwardRd[length - 1];
	float backwardBack = backwardRd[0];
	for (int i = 1; i < length; i++) {
		int j = i - 1;
		const float& forwardIn = forwardRd[j];
		const float& backwardIn = backwardRd[i];
		float& forwardOut = forwardWr[i];
		float& backwardOut = backwardWr[j];

		float k = scattering[j];

		forwardOut = ((1 + k) * forwardIn - k * backwardIn) * loss;
		backwardOut = ((1 - k) * backwardIn + k * forwardIn) * loss;
	}
	forwardWr[0] = (backwardBack + input) * loss;
	backwardWr[length - 1] = (forwardBack * -1) * loss;
	return forwardBack;
}

void SpeechEngineArticulator::TractSegment::stepBuffer(float* buf) {
}
#pragma endregion

#pragma region Tract
float SpeechEngineArticulator::VocalTract::step(float input) {
	return ts1.step(input);
}
#pragma endregion

void SpeechEngineArticulator::_init() {
	initLogger();
    logger.log("Type: Articulator", Logger::INFO);

#pragma region Animations
	_initArticulators();
	const PhonemeSet& ips = G_PS;
	for (int i = 0; i < ips.size(); i++) {
		const std::string& name = ips.xSampa(i);
		articAnim.loadGroup(std::format("configs/animations/phonemes/{}_anim.json", name), name);
	}
	articAnim.finalize();
	for (int i = 0; i < ips.size(); i++) {
		const std::string& name = ips.xSampa(i);
		phonToAnim[i] = articAnim.getAnimation(name);
	}
	// Disable animations
	systemPressure().animIndex = -1;

	// Set default anim to silence
	size_t initPhoneme = ips.fromString(L"spn");
	articAnim.startAnimation(initPhoneme);
#pragma endregion

	// Wavetable for a glottal pulse shape
	wavetable = new float[sampleRate];
	{
		double a = 0.1;
		float* tmpWavetable = new float[sampleRate];
		for (int i = 0; i < sampleRate; i++) {
			double x = (double)i / (sampleRate - 1);
			double fx = 5 * (x + 0.28);
			double sigma = 0;
			for (int k = 1; k < 100; k++) {
				sigma += (std::sin(k * fx)) / (k * k);
			}
			sigma = std::max(0.0, -sigma);
			tmpWavetable[i] = sigma;
		}
		// Smoothing
		int r = 300;
		for (int i = 0; i < sampleRate; i++) {
			float sum = 0;
			int count = 0;
			for (int w = 0; w < 1 + 2 * r; w++) {
				int index = sampleRate + i - r + w;
				sum += tmpWavetable[index % sampleRate];
				count++;
			}
			wavetable[i] = sum / count;
		}
		delete[] tmpWavetable;
	}

	logger.log("Initalized", Logger::VERBOSE);

	vocalTract = VocalTract(1011);
}

void SpeechEngineArticulator::_initArticulators() {
	pressure() = SpeechArticulator("configs/articulators/pressure.json", Animator::PRESSURE);
	systemPressure() = SpeechArticulator("configs/articulators/system_pressure.json", Animator::SYSTEM_PRESSURE);
	baseFrequency() = SpeechArticulator("configs/articulators/base_frequency.json", Animator::BASE_FREQUENCY);
	tongueBase() = SpeechArticulator("configs/articulators/tongue_base.json", Animator::TONGUE_BASE);
	tongueMid() = SpeechArticulator("configs/articulators/tongue_mid.json", Animator::TONGUE_MID);
	tongueTip() = SpeechArticulator("configs/articulators/tongue_tip.json", Animator::TONGUE_TIP);
	jaw() = SpeechArticulator("configs/articulators/jaw.json", Animator::JAW);
	lip() = SpeechArticulator("configs/articulators/lip.json", Animator::LIP);
}
