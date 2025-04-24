#include "PhonemeCollection.hpp"

#include <iostream>
#include "Global.hpp"
#include "Util.hpp"

int PhonemeCollection::getId(const std::string& name) {
  for (const PhonemeSet ps : phonemeSets)
    if (ps.name == name)
      return ps.id;
  G_LG(Util::format("Phoneme set with name '%s' does not exist", name.c_str()), Logger::DBUG);
  return -1;
}

std::string PhonemeCollection::getName(int id) {
  return get(id).name;
}

const PhonemeSet& PhonemeCollection::get(int id) {
  return phonemeSets[id];
}

const PhonemeSet& PhonemeCollection::get(const std::string& name) {
  return phonemeSets[getId(name)];
}

void PhonemeCollection::initPhonemeSet(std::string path) {
  JSONHelper json;
  if (json.open(path, -1, false)) {
        PhonemeSet ps(json["name"].get_string(), phonemeSets.size());
    
        // Map whatever symbols are used to an xsampa representation
    JSONHelper::JSONObj mappings = json["mappings"];
    addMappingToSet(mappings, ps);
    phonemeSets.push_back(std::move(ps));
    json.close();
  } else {
    G_LG(Util::format("Failed to open phoneme set at %s", path.c_str()), Logger::DEAD);
    throw;
  }
}

void PhonemeCollection::addMappingToSet(JSONHelper::JSONObj& group, PhonemeSet& ps) {
  size_t groupSize = group.get_array_size();
  for (size_t j = 0; j < groupSize; j++) {
    JSONHelper::JSONObj item = group[j];
    std::string token = item["token"].get_string(); // Token is the input from whatever source to be mapped from

    JSONHelper::JSONObj xSampa = item["x_sampa"];
    PhonemeSet::Phoneme phoneme;
    phoneme.symbol = xSampa["symbol"].get_string();

    ps.addString(token, phoneme.symbol);
  }
}

PhonemeCollection::PhonemeCollection() {
  JSONHelper json;
  if (json.open("configs/phoneme-collections.json", -1, false)) {
    JSONHelper::JSONObj sets = json["sets"];
    size_t nSets = sets.get_array_size();
    for (size_t i = 0; i < nSets; i++) {
      JSONHelper::JSONObj set = sets[i];
      std::string setName = set["name"].get_string();
      G_LG(Util::format("Initalizing phoneme set '%s'", setName.c_str()), Logger::DBUG);
      initPhonemeSet(set["path"].get_string());
    }
  } else {
    G_LG("Failed to open phoneme collections", Logger::DEAD);
    throw;
  }
}
