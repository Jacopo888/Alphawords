# AlphaScrabble: AlphaZero-style Scrabble Engine

A complete implementation of a Scrabble engine using Monte Carlo Tree Search (MCTS) with neural network guidance, inspired by AlphaZero but adapted for Scrabble gameplay.

## 🎯 Features

- **Complete Scrabble Rules**: Official English Scrabble rules with 15×15 board, premium squares, bingo bonuses
- **GADDAG/DAWG Lexicon**: Real lexicon support using Quackle's compiled dictionaries
- **Neural Network**: Policy + Value network architecture with CNN backbone
- **MCTS with PUCT**: Monte Carlo Tree Search with neural network guidance
- **Self-Play Training**: Generate training data through self-play games
- **CLI Interface**: Command-line tools for training, evaluation, and gameplay
- **Web Interface**: Interactive web application for playing against AI
- **Production Ready**: Docker deployment with Nginx, Redis, and PostgreSQL
- **Google Colab Ready**: Complete notebook for GPU training in Colab Pro
- **Comprehensive Tests**: Full test suite with pytest

## 🚀 Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alphascrabble/alphascrabble/blob/main/colab/AlphaScrabble_Colab.ipynb)

1. **Open the Colab notebook**: [AlphaScrabble_Colab.ipynb](colab/AlphaScrabble_Colab.ipynb)
2. **Run all cells**: The notebook will automatically:
   - Install all dependencies
   - Create the package structure
   - Test all components
   - Run performance benchmarks
3. **Start coding**: Everything is ready to use!

**Note**: The notebook creates a self-contained version of AlphaScrabble that works immediately in Colab without requiring external dependencies.

See [colab/COLAB_SETUP.md](colab/COLAB_SETUP.md) for detailed setup instructions.

### Web Interface

Start the web application for interactive gameplay:

```bash
# Start web server
./scripts/start_web.sh

# Open browser to http://localhost:5000
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/alphascrabble/alphascrabble.git
cd alphascrabble

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

### Production Deployment

Deploy to production with Docker:

```bash
# Deploy to production
./scripts/deploy.sh

# Access at http://localhost
```

## 📁 Project Structure

```
alphascrabble/
├── alphascrabble/           # Main package
│   ├── rules/              # Scrabble rules and game logic
│   ├── engine/             # MCTS, move generation, features
│   ├── nn/                 # Neural network models and training
│   ├── lexicon/            # GADDAG/DAWG lexicon interface
│   ├── training/           # Self-play training pipeline
│   ├── utils/              # Utilities (logging, I/O, etc.)
│   └── cli.py              # Command-line interface
├── web/                    # Web interface
│   ├── app.py              # Flask web application
│   ├── templates/          # HTML templates
│   └── requirements.txt    # Web dependencies
├── cpp/                    # C++ wrapper for Quackle
├── colab/                  # Google Colab notebook
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── docker-compose.prod.yml # Production deployment
├── nginx.conf              # Nginx configuration
├── third_party/            # Quackle source code
└── lexica_cache/           # Compiled lexicon files
```

## 🎮 Usage

### Command Line Interface

```bash
# Self-play to generate training data
alphascrabble selfplay --games 100 --out data/selfplay --simulations 160

# Train the neural network
alphascrabble train --data data/selfplay --epochs 10 --batch-size 256

# Evaluate against different opponents
alphascrabble eval --net-a checkpoints/best_model.pt --opponent random --games 50
alphascrabble eval --net-a checkpoints/best_model.pt --opponent greedy --games 50

# Play interactively against the bot
alphascrabble play --net checkpoints/best_model.pt --human-first
```

### Web Interface

The web interface provides an interactive way to play against the AI:

```bash
# Start web server
./scripts/start_web.sh

# Open browser to http://localhost:5000
```

Features:
- **Interactive Board**: Click to place tiles
- **AI Opponent**: Play against the trained neural network
- **Move Suggestions**: Get AI recommendations
- **Real-time Scoring**: Automatic score calculation
- **Game Statistics**: Track performance and progress

### Python API

```python
from alphascrabble import Board, GameState, AlphaScrabbleNet, MCTS
from alphascrabble.lexicon.gaddag_loader import GaddagLoader
from alphascrabble.engine.movegen import MoveGenerator

# Load lexicon
loader = GaddagLoader("lexica_cache/english_enable1.dawg", 
                     "lexica_cache/english_enable1.gaddag")
loader.load()

# Create game components
board = Board()
move_generator = MoveGenerator(loader)
model = AlphaScrabbleNet.load("checkpoints/best_model.pt")

# Initialize MCTS
mcts = MCTS(move_generator, feature_extractor, model)

# Play a game
game_state = GameState(...)
best_move = mcts.get_best_move(game_state)
```

## 🧠 Architecture

### Neural Network

- **Backbone**: CNN with 64 channels, 8 residual blocks
- **Input**: 32-channel board representation (letters, premiums, etc.)
- **Policy Head**: Per-move features → move probabilities
- **Value Head**: Board + rack embedding → position evaluation
- **Training**: Combined loss (policy + value + L2 regularization)

### MCTS Algorithm

- **Selection**: UCB1 with neural network priors
- **Expansion**: Generate legal moves using GADDAG
- **Evaluation**: Neural network policy + value
- **Backpropagation**: Update visit counts and values

### Move Generation

- **GADDAG**: Efficient word generation from anchor squares
- **DAWG**: Fast word validation
- **Features**: Score, premium usage, leave analysis, etc.

## 📊 Performance

### Training Requirements

- **GPU**: NVIDIA T4/A100 (Colab Pro) or equivalent
- **RAM**: 8GB+ recommended
- **Storage**: 2GB for lexicon + checkpoints
- **Time**: ~1 hour for 100 self-play games + training

### Model Performance

- **Parameters**: ~2M (lightweight for fast inference)
- **Inference**: ~10ms per move (GPU)
- **Training**: ~2 hours for 10 epochs on 1000 games

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_rules.py -v          # Game rules
pytest tests/test_movegen.py -v        # Move generation
pytest tests/test_mcts_basic.py -v     # MCTS algorithm
pytest tests/test_end2end_smoke.py -v  # End-to-end tests

# Run with coverage
pytest tests/ --cov=alphascrabble --cov-report=html
```

## 🔧 Configuration

Key configuration options in `alphascrabble/config.py`:

```python
# MCTS settings
MCTS_SIMULATIONS = 160    # Number of MCTS simulations per move
CPUCT = 1.5              # Exploration constant
TEMPERATURE = 1.0        # Move selection temperature

# Training settings
BATCH_SIZE = 256         # Training batch size
LEARNING_RATE = 0.001    # Adam learning rate
SELFPLAY_GAMES = 1000    # Games per self-play session

# Neural network
HIDDEN_CHANNELS = 64     # CNN channels
NUM_BLOCKS = 8          # Residual blocks
DROPOUT = 0.1           # Dropout rate
```

## 📚 Lexicon Setup

The system uses real Scrabble lexicons compiled from ENABLE1:

1. **Download**: ENABLE1 word list (172,000+ words)
2. **Compile**: Convert to DAWG format using Quackle
3. **Generate**: Create GADDAG for efficient move generation
4. **Verify**: Check lexicon integrity and word counts

Files are automatically downloaded and compiled in Colab, or manually:

```bash
# Download ENABLE1
wget https://norvig.com/ngrams/enable1.txt -O lexica_cache/enable1.txt

# Compile with Quackle
./third_party/quackle/build/src/makedawg lexica_cache/enable1.txt english.quackle_alphabet > lexica_cache/english_enable1.dawg
./third_party/quackle/build/src/makegaddag lexica_cache/english_enable1.dawg > lexica_cache/english_enable1.gaddag
```

## 🎯 Scrabble Rules

### Official English Scrabble

- **Board**: 15×15 grid with premium squares
- **Tiles**: 100 tiles (98 letters + 2 blanks)
- **Scoring**: A1, B3, C3, D2, E1, F4, G2, H4, I1, J8, K5, L1, M3, N1, O1, P3, Q10, R1, S1, T1, U1, V4, W4, X8, Y4, Z10, Blank 0
- **Bonuses**: Double/Triple Letter/Word scores, Bingo (+50)
- **Endgame**: Penalty for remaining tiles

### Premium Squares

- **Center**: First move must use center square
- **TWS**: Triple Word Score (8 positions)
- **DWS**: Double Word Score (16 positions)  
- **TLS**: Triple Letter Score (12 positions)
- **DLS**: Double Letter Score (24 positions)

## 🚀 Advanced Usage

### Custom Training

```python
from alphascrabble.nn.train import Trainer
from alphascrabble.nn.dataset import TrainingDataManager

# Load training data
data_manager = TrainingDataManager("data")
games = data_manager.load_selfplay_data("latest_selfplay.pkl")

# Configure training
train_config = {
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'value_loss_weight': 1.0,
    'policy_loss_weight': 1.0,
    'l2_weight': 1e-4
}

# Train model
trainer = Trainer(model, train_config, data_manager)
history = trainer.train(games[:800], games[800:], epochs=10)
```

### Model Evaluation

```python
from alphascrabble.nn.evaluate import Evaluator

# Create evaluator
evaluator = Evaluator(model, move_generator, feature_extractor, config)

# Evaluate against different opponents
random_results = evaluator.evaluate_against_random(num_games=100)
greedy_results = evaluator.evaluate_against_greedy(num_games=100)

# Compare models
previous_model = AlphaScrabbleNet.load("checkpoints/previous_model.pt")
comparison_results = evaluator.evaluate_against_previous(previous_model, num_games=50)
```

### Custom Move Generation

```python
from alphascrabble.engine.movegen import MoveGenerator

# Generate moves for specific position
moves = move_generator.generate_moves(board, rack)

# Filter moves by criteria
high_score_moves = [m for m in moves if m.total_score > 20]
bingo_moves = [m for m in moves if m.is_bingo]
premium_moves = [m for m in moves if any(
    board.get_premium_type(tile.position.row, tile.position.col) != PremiumType.NONE
    for tile in m.tiles
)]
```

## 🐛 Troubleshooting

### Common Issues

1. **qlex module not found**: C++ wrapper not compiled
   ```bash
   pip install -e .  # Reinstall to compile C++ extension
   ```

2. **Lexicon files missing**: Download and compile ENABLE1
   ```bash
   # Run lexicon setup in Colab or manually compile
   ```

3. **CUDA out of memory**: Reduce batch size or model size
   ```python
   config.BATCH_SIZE = 128  # Reduce from 256
   ```

4. **Slow move generation**: Increase MCTS simulations or use GPU
   ```python
   config.MCTS_SIMULATIONS = 320  # Increase from 160
   ```

### Performance Optimization

- **GPU Training**: Use CUDA-enabled PyTorch
- **Batch Inference**: Process multiple positions simultaneously
- **Model Quantization**: Use INT8 for faster inference
- **Caching**: Cache move generation results

## 📈 Results

### Training Progress

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics:
- **Policy Loss**: Cross-entropy between MCTS and neural network policies
- **Value Loss**: MSE between game outcomes and value predictions
- **Win Rate**: Performance against random/greedy opponents

### Expected Performance

- **vs Random**: 95%+ win rate
- **vs Greedy**: 70-80% win rate  
- **vs Previous Model**: 55-65% win rate (improvement)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### GitHub Actions Setup

The CI/CD pipeline includes multiple workflows:

- **✅ Simple CI**: Basic testing (always works)
- **✅ Test Suite**: Comprehensive testing with pytest
- **✅ Build**: Package and Docker building
- **⚠️ Deploy**: Docker Hub deployment (requires secrets)
- **⚠️ Release**: PyPI releases (requires secrets)

**Current Status**: Basic workflows are active and working. Advanced features require GitHub secrets.

See [.github/WORKFLOWS.md](.github/WORKFLOWS.md) for workflow details and [.github/SECRETS.md](.github/SECRETS.md) for secrets setup.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black alphascrabble/
isort alphascrabble/
flake8 alphascrabble/

# Run type checking
mypy alphascrabble/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Quackle**: Scrabble engine and lexicon tools
- **ENABLE1**: Comprehensive English word list
- **AlphaZero**: MCTS + neural network inspiration
- **PyTorch**: Deep learning framework
- **Google Colab**: Free GPU access for development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alphascrabble/alphascrabble/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alphascrabble/alphascrabble/discussions)
- **Documentation**: [Wiki](https://github.com/alphascrabble/alphascrabble/wiki)

---

**Happy Scrabble playing! 🎲📚**

*Built with ❤️ for the Scrabble and AI communities*