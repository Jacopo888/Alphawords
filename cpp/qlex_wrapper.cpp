#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <memory>

// Quackle includes
#include "lexicon.h"
#include "gaddag.h"
#include "board.h"
#include "move.h"
#include "tile.h"

namespace py = pybind11;

class QLexWrapper {
private:
    std::unique_ptr<Quackle::Lexicon> dawg_;
    std::unique_ptr<Quackle::Gaddag> gaddag_;
    bool loaded_;

public:
    QLexWrapper() : loaded_(false) {}
    
    void load_lexica(const std::string& dawg_path, const std::string& gaddag_path) {
        try {
            dawg_ = std::make_unique<Quackle::Lexicon>();
            gaddag_ = std::make_unique<Quackle::Gaddag>();
            
            dawg_->load(dawg_path);
            gaddag_->load(gaddag_path);
            
            loaded_ = true;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load lexica: " + std::string(e.what()));
        }
    }
    
    bool is_loaded() const {
        return loaded_;
    }
    
    bool is_word(const std::string& word) const {
        if (!loaded_) return false;
        return dawg_->isWord(word);
    }
    
    std::vector<std::string> generate_moves(
        const std::string& board_state,
        const std::string& rack
    ) {
        if (!loaded_) {
            throw std::runtime_error("Lexica not loaded");
        }
        
        // This is a simplified implementation
        // In practice, you'd need to parse board_state and use GADDAG
        std::vector<std::string> moves;
        
        // Placeholder: return some basic moves
        moves.push_back("HELLO A8");
        moves.push_back("WORLD A9");
        
        return moves;
    }
    
    int get_word_count() const {
        if (!loaded_) return 0;
        return dawg_->size();
    }
};

PYBIND11_MODULE(qlex, m) {
    m.doc() = "Quackle lexicon wrapper for AlphaScrabble";
    
    py::class_<QLexWrapper>(m, "QLexWrapper")
        .def(py::init<>())
        .def("load_lexica", &QLexWrapper::load_lexica)
        .def("is_loaded", &QLexWrapper::is_loaded)
        .def("is_word", &QLexWrapper::is_word)
        .def("generate_moves", &QLexWrapper::generate_moves)
        .def("get_word_count", &QLexWrapper::get_word_count);
}
