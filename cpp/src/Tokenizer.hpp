#pragma once
#include <fstream>
#include <vector>
#include <string>

class Tokenizer {
public:
    Tokenizer() = default;

    ~Tokenizer() = default;

    inline int vocab_size() const {
        return m_n_vocab;
    }

    void load(const std::string& token_txt) {
        std::ifstream ifs(token_txt);
        std::string line;

        while (std::getline(ifs, line)) {
            m_tokens.push_back(line);
        }
    }

    std::string id_to_piece(int token) {
        return m_tokens[token];
    }

private:
    std::vector<std::string> m_tokens;
    int m_n_vocab;
}