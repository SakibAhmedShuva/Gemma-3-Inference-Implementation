# Gemma 3 Inference Implementation

## Overview
This repository contains code for running inference with Google's Gemma 3 1B Instruct model using Hugging Face Transformers. The implementation demonstrates both standard and streaming inference approaches.

## Features
- Authentication with Hugging Face Hub
- Basic inference with Gemma 3 1B Instruct model
- Chat history management
- Streaming text generation with real-time output
- GPU acceleration with automatic device detection

## Requirements
- Python 3.x
- Hugging Face Transformers
- PyTorch
- CUDA-compatible GPU (optional, for faster inference)

## Usage
The notebook provides three main implementations:

1. **Basic inference**:
   ```python
   response, chat_history = chat_with_gemma("Your message here")
   ```

2. **Streaming inference**:
   ```python
   response, chat_history = chat_with_gemma_streaming("Your message here")
   ```

3. **GPU-accelerated streaming inference**:
   ```python
   # Automatically uses GPU if available
   response, chat_history = chat_with_gemma_streaming("Your message here")
   ```

## Notes
- The model requires authentication with Hugging Face Hub
- Streaming implementation uses threading for non-blocking generation
- GPU implementation uses half-precision (float16) when running on CUDA
- Chat history is maintained across interactions for contextual responses

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
