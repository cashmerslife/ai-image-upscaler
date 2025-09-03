# 🚀 AI Image Upscaler

AI-powered image upscaler using Stable Diffusion XL for high-quality image enhancement and super-resolution.

## ✨ Features

- **High-Quality Upscaling**: Uses Stable Diffusion XL for superior image enhancement
- **Flexible Scale Factors**: Support for 1x to 4x upscaling
- **Prompt Control**: Optional prompts and negative prompts for guided enhancement
- **Advanced Parameters**: Fine-tune creativity, resemblance, and inference steps
- **Multiple Formats**: Supports common image formats (JPEG, PNG, etc.)
- **Web Interface**: Easy-to-use Gradio interface
- **Docker Support**: Containerized deployment ready

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended) or CPU
- Git LFS (for model files)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cashmerslife/ai-image-upscaler.git
   cd ai-image-upscaler
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (typically `http://localhost:7860`)

### Docker Installation

1. **Build the Docker image:**
   ```bash
   docker build -t ai-image-upscaler .
   ```

2. **Run the container:**
   ```bash
   docker run -p 7860:7860 ai-image-upscaler
   ```

3. **Access the application** at `http://localhost:7860`

## 🎯 Usage

### Basic Usage

1. **Upload an image** using the file upload interface
2. **Set the scale factor** (1x to 4x)
3. **Click "🚀 Upscale Image"** to process
4. **Download the enhanced image** from the output section

### Advanced Options

#### Prompts
- **Prompt**: Describe desired enhancements (e.g., "sharp, detailed, high quality")
- **Negative Prompt**: Specify what to avoid (e.g., "blurry, low quality, artifacts")

#### Parameters
- **Scale Factor**: How much to enlarge the image (1-4x)
- **Guidance Scale**: How closely to follow prompts (1-20)
- **Creativity**: How much to modify the original (0.1-1.0)
- **Resemblance**: How closely to match the original (0.1-1.0)
- **Inference Steps**: Quality vs speed trade-off (1-50)
- **Seed**: For reproducible results

#### Performance Options
- **Enable Downscaling**: Reduce large images before processing for speed
- **Downscaling Resolution**: Maximum dimension for downscaling (256-1024px)

## 📋 Requirements

- **gradio**: Web interface framework
- **torch**: PyTorch for deep learning
- **diffusers**: Stable Diffusion XL pipeline
- **transformers**: Model loading and processing
- **Pillow**: Image processing
- **numpy**: Numerical operations

See `requirements.txt` for complete dependency list.

## 🐳 Docker

The application includes a Dockerfile for easy deployment:

- **Base Image**: Python 3.9 slim
- **Port**: 7860
- **Environment**: Configured for Gradio hosting
- **Dependencies**: All requirements pre-installed

## 🔧 Configuration

### Environment Variables

- `GRADIO_SERVER_NAME`: Server host (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)

### Model Configuration

The application uses `stabilityai/stable-diffusion-xl-refiner-1.0` by default. Models are automatically downloaded on first use.

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production with Docker
```bash
docker build -t ai-image-upscaler .
docker run -d -p 7860:7860 ai-image-upscaler
```

### Cloud Deployment

The application is ready for deployment on:
- Hugging Face Spaces
- Google Cloud Run
- AWS ECS
- Azure Container Instances

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Built with ❤️ using Stable Diffusion XL and Gradio**
