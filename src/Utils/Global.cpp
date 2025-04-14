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
		.outputTo(Logger::WARN)
		.outputTo(Logger::ERRO)
		.outputTo(Logger::DEAD)
		.enableColor(true));
	logger.addStream(Logger::Stream("log.txt")
		.outputTo(Logger::DBUG)
		.outputTo(Logger::INFO)
		.outputTo(Logger::WARN)
		.outputTo(Logger::ERRO)
		.outputTo(Logger::DEAD));

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
			G_LG("No phoneme mapping groups found", Logger::WARN);
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
		G_LG("Failed to open phoneme mappings", Logger::DEAD);
		throw("Failed to open phoneme mappings");
	}
}

void Global::log(const std::string& message, int verbosity, int color) {
	logger.log(message, verbosity, color);
}
