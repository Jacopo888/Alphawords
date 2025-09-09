# Google Colab Setup Instructions

## 🚀 Quick Start

1. **Open the notebook**: [AlphaScrabble_Colab.ipynb](AlphaScrabble_Colab.ipynb)
2. **Run all cells**: Click "Runtime" → "Run all"
3. **Wait for completion**: The setup takes ~2-3 minutes
4. **Test the demo**: All components will be tested automatically

## 📋 What the Notebook Does

### 1. **Environment Setup**
- Installs system dependencies (Qt, build tools)
- Installs Python packages (PyTorch, NumPy, etc.)
- Sets up GPU support automatically

### 2. **Package Creation**
- Creates the `alphascrabble` package structure
- Implements core game components (Board, Tile, TileBag)
- Implements neural network architecture
- Implements feature extraction

### 3. **Testing & Demo**
- Tests all components individually
- Tests neural network inference
- Tests feature extraction
- Tests complete game setup
- Performance benchmarking

## 🔧 Troubleshooting

### **ModuleNotFoundError: No module named 'alphascrabble'**
**Solution**: The notebook creates the module automatically. Make sure you run all cells in order.

### **CUDA Out of Memory**
**Solution**: 
- Use Colab Pro for more GPU memory
- Reduce batch size in performance tests
- Restart runtime and run again

### **Installation Fails**
**Solution**:
- Check internet connection
- Restart runtime
- Run cells individually

### **Import Errors**
**Solution**:
- Ensure all cells ran successfully
- Check that `sys.path.append('.')` is executed
- Verify file creation in the file browser

## 📊 Expected Output

When everything works correctly, you should see:

```
✅ Dependencies installed successfully!
✅ AlphaScrabble package structure created!
✅ Rules module created!
✅ Neural network module created!
✅ Feature extraction module created!
🖥️  Using device: cuda
🚀 GPU: Tesla T4
💾 GPU Memory: 15.0 GB
✅ Board created: 15x15
✅ Tile bag created: 100 tiles
✅ Tiles created: A=1, B=3
✅ Tiles placed on board
✅ Model created with 1048576 parameters
✅ Board features: (32, 15, 15)
✅ Rack features: (27,)
✅ Forward pass successful!
📊 Policy logits shape: torch.Size([1, 64])
📊 Value shape: torch.Size([1, 1])
📊 Value range: [-0.123, 0.456]
✅ Prediction method: policy (1, 64), value 0.234
🎉 All components working correctly!
```

## 🎯 Performance Expectations

### **GPU (Colab Pro)**
- **Setup time**: ~2-3 minutes
- **Inference speed**: ~100+ samples/second
- **Feature extraction**: ~1000+ extractions/second
- **Memory usage**: ~2-4 GB GPU memory

### **CPU (Free Colab)**
- **Setup time**: ~3-5 minutes
- **Inference speed**: ~10-20 samples/second
- **Feature extraction**: ~100+ extractions/second
- **Memory usage**: ~1-2 GB RAM

## 🔄 Re-running the Notebook

If you need to re-run the notebook:

1. **Restart runtime**: Runtime → Restart runtime
2. **Run all cells**: Runtime → Run all
3. **Wait for completion**: All setup will be re-done

## 📁 File Structure Created

The notebook creates this structure:

```
/content/
├── alphascrabble/
│   ├── __init__.py
│   ├── rules/
│   │   ├── __init__.py
│   │   └── board.py
│   ├── nn/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── engine/
│   │   ├── __init__.py
│   │   └── features.py
│   ├── lexicon/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
└── AlphaScrabble_Colab.ipynb
```

## 🚀 Next Steps

After successful setup:

1. **Train the model**: Implement self-play training
2. **Add move generation**: Implement GADDAG/DAWG lexicon
3. **Implement MCTS**: Add Monte Carlo Tree Search
4. **Create UI**: Build web interface
5. **Deploy**: Use in production

## 💡 Tips

- **Use Colab Pro**: Better GPU and more memory
- **Save your work**: Download the notebook when done
- **Check GPU**: Ensure GPU is enabled in Runtime settings
- **Monitor memory**: Watch GPU memory usage in performance tests
- **Run incrementally**: Test each cell individually if needed

## 🆘 Getting Help

If you encounter issues:

1. **Check the output**: Look for error messages
2. **Restart runtime**: Often fixes temporary issues
3. **Run cells individually**: Isolate the problem
4. **Check GPU**: Ensure GPU is enabled
5. **Verify internet**: Ensure stable connection

The notebook is designed to be self-contained and should work out of the box! 🎉
