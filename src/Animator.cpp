#include "Animator.hpp"

#include <assert.h>
#include "ClassifierHelper.hpp"

float Animator::CubicBezierCurve::pos(float t) {
	assert(0 <= t && t <= 1);
	float b1, b2, b3, c1, c2;

	b1 = std::lerp(p1, p2, t);
	b2 = std::lerp(p2, p3, t);
	b3 = std::lerp(p3, p4, t);

	c1 = std::lerp(b1, b2, t);
	c2 = std::lerp(b2, b3, t);

	return std::lerp(c1, c2, t);
}

float Animator::Curve::pos(float t) {
	assert(subCurves.size() > 0);
	for (CubicBezierCurve& sc : subCurves) {
		if (sc.start <= t && t <= sc.end) {
			float subt = t - sc.start;
			subt /= sc.length;
			return sc.pos(subt);
		}
	}
	return subCurves.back().p4;
}

Animator::Curve::Curve() {
}

Animator::Curve::Curve(const JSONHelper::JSONObj& obj) {
	int size = obj.get_array_size();
	assert(size > 0);
	for (int i = 0; i < size; i++) {
		JSONHelper::JSONObj item = obj[i];
		CubicBezierCurve sc;
		sc.start = item["start"].get_real();
		sc.end = item["end"].get_real();
		sc.length = sc.end - sc.start;
		sc.p1 = item["p1"].get_real();
		sc.p2 = item["p2"].get_real();
		sc.p3 = item["p3"].get_real();
		sc.p4 = item["p4"].get_real();
		subCurves.push_back(sc);
	}
}

int Animator::getAnimation(const std::string& name) {
	assert(_initDone);
	for (int i = 0; i < animNames.size(); i++) {
		if (animNames[i] == name)
			return i;
	}
	assert(false);
	return -1;
}

void Animator::startAnimation(int animGroup) {
	assert(0 <= animGroup && animGroup < animationGroups.size());
	assert(_initDone);
	time = 0;
	currentGroup = animGroup;
}

float Animator::timeSinceStart() {
	assert(_initDone);
	return time;
}

void Animator::stepTime(float t) {
	assert(_initDone);
	time += t;
}

float Animator::getAnimationPos(int articulator) {
	assert(_initDone);
	assert(0 <= articulator && articulator < articulatorsList.size());
	float pos = animationGroups[currentGroup][articulator].pos(time);
	assert(std::isfinite(pos));
	return pos;
}

int Animator::getArticulator(const std::string& name) {
	if (name == "pressure") {
		return PRESSURE;
	} else if (name == "system_pressure") {
		return SYSTEM_PRESSURE;
	} else if (name == "base_frequency") {
		return BASE_FREQUENCY;
	} else if (name == "tongue_base") {
		return TONGUE_BASE;
	} else if (name == "tongue_mid") {
		return TONGUE_MID;
	} else if (name == "tongue_tip") {
		return TONGUE_TIP;
	} else if (name == "jaw") {
		return JAW;
	} else if (name == "lip") {
		return LIP;
	} else {
		assert(false);
		return -1;
	}
}

bool Animator::loadGroup(const std::string& path, const std::string& name) {
	assert(!_initDone);
	JSONHelper json;
	if (!json.open(path, _jsonVersion)) {
		json["name"] = name;
		for (const std::string& articulator : articulatorsList) {
			JSONHelper::JSONObj animObj = json[articulator.c_str()];
			JSONHelper::JSONObj animArr = animObj.add_arr("animation");
			for (int i = 0; i < 1; i++) {
				JSONHelper::JSONObj subCurve = animArr.append();
				subCurve["start"] = 0.0;
				subCurve["end"] = 1.0;
				subCurve["p1"] = 0.0;
				subCurve["p2"] = 0.0;
				subCurve["p3"] = 0.0;
				subCurve["p4"] = 0.0;
			}
		}
		json.save();
		return false;
	}

	assert(json["name"].get_string() == name);
	int group = animNames.size();
	animNames.push_back(name);
	animationGroups.push_back(std::vector<Curve>(articulatorsList.size()));
	for (const std::string& articulator : articulatorsList) {
		JSONHelper::JSONObj animObj = json[articulator.c_str()];
		JSONHelper::JSONObj animArr = animObj["animation"];
		animationGroups[group][getArticulator(articulator)] = Curve(animArr);
	}
	return true;
}

void Animator::finalize() {
	assert(!_initDone);
	_initDone = true;
}
