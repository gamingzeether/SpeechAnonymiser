#pragma once

#include "../common_inc.hpp"

#include <vector>
#include "JSONHelper.hpp"
#include "PhonemeSet.hpp"

// Access phoneme sets through Global::get().getPhonemeSet(index)
// Or the macro G_PS defined in Global.hpp
class PhonemeCollection {
public:
    int getId(const std::string& name);
    std::string getName(int id);
    const PhonemeSet& get(int id);
    const PhonemeSet& get(const std::string& name);
private:
    std::vector<PhonemeSet> phonemeSets;

    void initPhonemeSet(std::string path);
    void addMappingToSet(JSONHelper::JSONObj& group, PhonemeSet& ps);

    PhonemeCollection();

    friend class Global;
};
