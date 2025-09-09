# AlphaScrabble Colab

This directory contains files specifically for running AlphaScrabble in Google Colab.

## Files

- `AlphaScrabble_Colab.ipynb` - Main Colab notebook with complete demo
- `install.sh` - Installation script for Colab environment
- `demo.py` - Demo script to test basic functionality
- `quick_start.py` - Quick start script for getting started
- `requirements.txt` - Colab-specific requirements

## Quick Start in Colab

1. **Open the notebook**: [AlphaScrabble_Colab.ipynb](AlphaScrabble_Colab.ipynb)
2. **Run all cells**: The notebook will automatically install everything
3. **Follow the demo**: Each cell demonstrates a different aspect of AlphaScrabble

## Manual Installation

If you prefer to install manually:

```bash
# Run the installation script
!bash colab/install.sh

# Or run the quick start
!python colab/quick_start.py
```

## Features Demonstrated

- ✅ **Setup**: Automatic dependency installation
- ✅ **Lexicon**: ENABLE1 download and compilation
- ✅ **Neural Network**: Model creation and testing
- ✅ **Self-Play**: Training data generation
- ✅ **Training**: Neural network training
- ✅ **Evaluation**: Model performance testing
- ✅ **Interactive Play**: Gameplay demonstration

## Requirements

- Google Colab Pro (recommended for GPU access)
- Python 3.10+
- CUDA-compatible GPU (optional but recommended)

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or model size
2. **Slow execution**: Use GPU runtime in Colab
3. **Import errors**: Make sure to run `pip install -e .` first
4. **Lexicon errors**: Run the lexicon setup script

### Performance Tips

- Use GPU runtime for faster training
- Increase MCTS simulations for stronger play
- Use larger batch sizes for faster training
- Save checkpoints regularly

## Next Steps

After running the Colab notebook:

1. **Local Installation**: Install AlphaScrabble locally for development
2. **Custom Training**: Modify the training parameters
3. **Model Evaluation**: Test against different opponents
4. **Interactive Play**: Play games against the trained model

## Support

- **Issues**: [GitHub Issues](https://github.com/alphascrabble/alphascrabble/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alphascrabble/alphascrabble/discussions)
- **Documentation**: [Main README](../README.md)
