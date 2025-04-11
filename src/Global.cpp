#include "Global.hpp"

#include "ClassifierHelper.hpp"
#include "JSONHelper.hpp"
#include "Util.hpp"

size_t Global::silencePhone() {
	return silPhone;
}

Global::Global() {
	logger.addStream(Logger::Stream(std::cout)
		.outputTo(Logger::INFO)
		.outputTo(Logger::WARNING)
		.outputTo(Logger::ERR)
		.outputTo(Logger::FATAL)
		.enableColor(true));
	logger.addStream(Logger::Stream("global.log")
		.outputTo(Logger::VERBOSE)
		.outputTo(Logger::INFO)
		.outputTo(Logger::WARNING)
		.outputTo(Logger::ERR)
		.outputTo(Logger::FATAL));

	initPhonemeSet();
	silPhone = ps.fromString("");
}

void Global::initPhonemeSet() {
	// X-Sampa for this group of phoneme tokens
	JSONHelper json;
	if (json.open("configs/phoneme_maps.json")) {
		JSONHelper::JSONObj mappings = json["mappings"];
		size_t arrSize = mappings.get_array_size();
		if (arrSize == 0) {
			logger.log("No phoneme mapping groups found", Logger::WARNING);
		}
		for (size_t i = 0; i < arrSize; i++) {
			JSONHelper::JSONObj group = mappings[i];
			size_t groupSize = group.get_array_size();
			for (size_t j = 0; j < groupSize; j++) {
				JSONHelper::JSONObj item = group[j];
				std::string token = item["token"].get_string();
				JSONHelper::JSONObj xSampa = item["x_sampa"];

				PhonemeSet::Phoneme phoneme;
				phoneme.symbol = xSampa["symbol"].get_string();
				if (xSampa.exists("diacritic")) {
					phoneme.diacritic = xSampa["diacritic"].get_string();
				}

				if (i > 0 && !ps.xSampaExists(phoneme.symbol)) {
					std::printf("Does not exists: %s\n", phoneme.symbol.c_str());
				}

				ps.addString(token, phoneme.symbol);
			}
		}
		json.close();
	} else {
		logger.log("Failed to open phoneme mappings", Logger::FATAL);
		throw("Failed to open phoneme mappings");
	}
}
