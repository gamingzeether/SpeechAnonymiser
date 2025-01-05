#pragma once

#include "common_inc.h"

#include <string>
#include <vector>
#include <map>
#include "structs.h"
#include "Config.h"

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
		size_t from;
		std::vector<size_t> glide;
		size_t to;
	};
	struct Unit {
		size_t index;
		Features features;
		std::string alias;
		std::vector<float> audio;
		bool loaded;
		uint32_t consonant;
		uint32_t preutterance;
		uint32_t overlap;

		void save(int sr, const std::string& cacheDir) const;
		void unload();
		void load(const std::string& cacheDir);
	};
	struct DesiredFeatures {
		const Unit* from;
		std::vector<size_t> glide;
		size_t to;
	};

	Voicebank() {};
	Voicebank& targetSamplerate(int sr);
	Voicebank& setShort(const std::string& name);
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
	void loadAliases();

	Config config;
	std::string directory;
	std::string cacheDirectory;
	std::string shortName = "";
	int samplerate;
	std::vector<Unit> units;
	std::map<std::string, Features> aliasMapping;
};
