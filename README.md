# Text-to-Image Generator

A Python application that uses Stable Diffusion to generate images from text prompts.

![Example Generated Image](/api/placeholder/512/256)

## Features

- Generate high-quality images from text descriptions
- Interactive web UI powered by Gradio
- Command-line interface for batch processing
- Customize generation parameters (dimensions, steps, guidance scale)
- Support for negative prompts to avoid unwanted elements
- Multi-image generation with grid view
- Seed control for reproducible results

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.12+
- 8GB+ VRAM for GPU acceleration (optional but recommended)

### Setup

1. Clone this repository:
   bash
   git clone https://github.com/yourusername/text-to-image-generator.git
   cd text-to-image-generator
   

2. Install the required dependencies:
   bash
   pip install torch diffusers transformers gradio pillow
   

3. (Optional) For GPU acceleration, install CUDA-compatible PyTorch:
   bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
   

## Usage

### Web Interface

Launch the Gradio web interface:

bash
python text_to_image_generator.py


Then open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).

### Command Line

Generate a single image:

bash
python text_to_image_generator.py --prompt "A magical forest with glowing mushrooms and a small cottage"


Generate multiple images:

bash
python text_to_image_generator.py --prompt "Cyberpunk cityscape" --num_images 4


### Advanced Options

bash
python text_to_image_generator.py \
  --prompt "A beautiful beach scene" \
  --negative_prompt "people, cloudy, dark, rain" \
  --height 512 \
  --width 768 \
  --steps 30 \
  --guidance 7.5 \
  --num_images 2 \
  --seed 42 \
  --output_dir "my_images" \
  --model "runwayml/stable-diffusion-v1-5"


## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| --prompt | Text description of the image to generate | Required |
| --negative_prompt | Elements to avoid in the image | "" |
| --height | Height of the output image | 512 |
| --width | Width of the output image | 512 |
| --steps | Number of inference steps | 30 |
| --guidance | Guidance scale (how closely to follow the prompt) | 7.5 |
| --num_images | Number of images to generate | 1 |
| --seed | Random seed for reproducible results (-1 for random) | None |
| --output_dir | Directory to save output images | "outputs" |
| --model | Stable Diffusion model to use | "runwayml/stable-diffusion-v1-5" |
| --device | Device to use (cuda/cpu) | Auto-detected |

## API Usage

The TextToImageGenerator class can be imported and used in your own Python projects:

python
from text_to_image_generator import TextToImageGenerator

generator = TextToImageGenerator()
images = generator.generate_image(
    prompt="A futuristic city with flying cars",
    num_images=1
)
generator.save_images(images, output_dir="my_custom_output")


## Performance Tips

- Use a CUDA-compatible GPU for significantly faster generation
- Decrease the number of inference steps for faster results (at the cost of quality)
- Adjust image dimensions to match your needs (larger images require more VRAM)
- Use attention slicing to reduce memory usage (enabled by default)

## Project Structure


text-to-image-generator/
├── text_to_image_generator.py  # Main application code
├── outputs/                    # Generated images
├── README.md                   # This file
└── requirements.txt            # Dependencies


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the Stable Diffusion implementation
- [Stability AI](https://stability.ai/) for releasing Stable Diffusion
- [Gradio](https://gradio.app/) for the web interface

## Future Improvements

- Support for more Stable Diffusion models
- Image-to-image generation
- Inpainting and outpainting capabilities
- LoRA fine-tuning support
- Advanced prompt engineering tools
