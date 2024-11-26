#pragma once

#include "common_inc.h"

#include <string>
#include <vector>
#include <armadillo>
#include "structs.h"
#include "JSONHelper.h"

class Voicebank {
public:
	struct UTAULine {
		std::string file;
		std::string alias;
		float offset;
		float consonant;
		float cutoff;
		float preutterance;
		float overlap;
	};
	struct Features {

	};
	struct Unit {
		size_t index;
		Features features;
		std::vector<float> audio;
		bool loaded;

		void save(int sr, const std::string& cacheDir) const;
		void unload();
		void load(const std::string& cacheDir);
	};
	struct DesiredFeatures {
		Unit& from;
		std::vector<Phone> glide;
		Phone to;
	};

	Voicebank() {};
	Voicebank& targetSamplerate(int sr);
	Voicebank& open(const std::string& directory);

	const Unit& selectUnit(const DesiredFeatures& features);
	void loadUnit(size_t index);
	void unloadUnit(size_t index);
private:
	void loadUnits(const std::vector<UTAULine>& lines);
	std::string bankName();
	std::string cacheDir();
	bool isCached();
	void saveCache();
	bool loadCache();
	static float cost(const Features& f1, const Features& f2);

	JSONHelper json;
	std::string directory;
	std::string cacheDirectory;
	int samplerate;
	std::vector<Unit> units;
};