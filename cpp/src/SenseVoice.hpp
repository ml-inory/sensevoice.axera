#pragma once
#include <string>
#include <cassert>
#include "EngineWrapper.hpp"
#include "Tokenizer.hpp"
#include "WavFrontend.hpp"

class SenseVoice {
public:
    SenseVoice(
        const std::string& model_path,
        const std::string& token_txt,
        int max_len = 256,
        const std::string& language = "auto",
        bool use_itn = true
    ):
        m_max_len(max_len),
        m_language(language),
        m_use_itn(use_itn)
    {
        assert (0 == m_model.Init(model_path.c_str()));
        m_tokenizer.load(token_txt);
    }

    ~SenseVoice() {
        m_model.Release();
    }

private:
    EngineWrapper m_model;
    Tokenizer m_tokenizer;
    WavFrontend m_frontend;
    int m_max_len;
    std::string m_language;
    bool m_use_itn;
};