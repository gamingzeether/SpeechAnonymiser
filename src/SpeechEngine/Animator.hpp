#pragma once

#include "../common_inc.hpp"

#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include "../Utils/JSONHelper.hpp"

// Class for animating articulators in a SpeechEngine
class Animator {
public:
	enum Articulator {
		PRESSURE,
		SYSTEM_PRESSURE,
		BASE_FREQUENCY,
		TONGUE_BASE,
		TONGUE_MID,
		TONGUE_TIP,
		JAW,
		LIP,
	};

	int getAnimation(const std::string& name);
	int currentAnim() { return currentGroup; };
	void startAnimation(int animGroup);
	float timeSinceStart();
	void stepTime(float t);
	float getAnimationPos(int articulator);
	// Gets index of articulator with name; try to use enum instead
	int getArticulator(const std::string& name);
	bool loadGroup(const std::string& path, const std::string& name);
	// Prevent modifying and allow accessing
	void finalize();
private:
	struct CubicBezierCurve {
		float start, end, length;
		float p1, p2, p3, p4;
		float pos(float t);
	};

	class Curve {
	public:
		float pos(float t);

		Curve();
		Curve(const JSONHelper::JSONObj& obj);
	private:
		std::vector<CubicBezierCurve> subCurves = std::vector<CubicBezierCurve>();
	};

	bool _initDone = false;
	int _jsonVersion = 0;
	float time = 0;
	int currentGroup;
	std::vector<std::string> animNames = std::vector<std::string>();
	std::vector<std::vector<Curve>> animationGroups = std::vector<std::vector<Curve>>();
	const inline static std::vector<std::string> articulatorsList = { "pressure", "system_pressure", "base_frequency", "tongue_base", "tongue_mid", "tongue_tip", "jaw", "lip" };
};
