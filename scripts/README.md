# AlphaScrabble Scripts

This directory contains utility scripts for AlphaScrabble development and deployment.

## Scripts

### Development Scripts

- `dev_setup.sh` - Set up development environment
- `run_tests.sh` - Run test suite with coverage
- `benchmark.sh` - Run performance benchmarks
- `clean.sh` - Clean build artifacts and temporary files

### Deployment Scripts

- `setup_lexicon.sh` - Download and compile ENABLE1 lexicon

## Usage

### Development Setup

```bash
# Set up development environment
./scripts/dev_setup.sh

# Run tests
./scripts/run_tests.sh

# Run benchmarks
./scripts/benchmark.sh

# Clean build artifacts
./scripts/clean.sh
```

### Lexicon Setup

```bash
# Download and compile ENABLE1 lexicon
./scripts/setup_lexicon.sh
```

## Requirements

- Python 3.10+
- pip
- git
- cmake
- build-essential (Linux)
- qtbase5-dev (Linux)

## Environment Variables

- `VIRTUAL_ENV` - Virtual environment path (optional)
- `CUDA_VISIBLE_DEVICES` - GPU device selection (optional)

## Troubleshooting

### Common Issues

1. **Permission denied**: Make sure scripts are executable
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Missing dependencies**: Install system dependencies
   ```bash
   sudo apt-get update
   sudo apt-get install -y qtbase5-dev libqt5core5a build-essential cmake ninja-build
   ```

3. **Python version**: Ensure Python 3.10+ is installed
   ```bash
   python --version
   ```

4. **Virtual environment**: Use a virtual environment for development
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

### Performance Tips

- Use GPU for faster training and inference
- Increase batch size for better GPU utilization
- Use larger MCTS simulations for stronger play
- Cache feature extraction results when possible

## Contributing

When adding new scripts:

1. Make them executable: `chmod +x script_name.sh`
2. Add error handling with `set -e`
3. Include usage information
4. Add to this README
5. Test on different platforms

## License

Same as the main project (MIT License).
