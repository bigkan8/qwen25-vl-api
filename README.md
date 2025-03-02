# Qwen2.5-VL API for Render

This repository contains code to deploy the Qwen2.5-VL-7B-Instruct model on Render as an API service. This allows you to run image analysis and screen parsing without having to run the model locally.

## Features

- API endpoint for analyzing images with custom prompts
- Specialized endpoint for analyzing screens with parsed elements from OmniParser
- Support for 8-bit quantization to improve performance
- Health check endpoints
- JSON response extraction

## Deployment to Render

### Prerequisites

1. A [Render](https://render.com/) account
2. Git repository with this code

### Deployment Steps

1. **Fork or clone this repository** to your own GitHub account.

2. **Log in to Render** and go to your dashboard.

3. **Click on "New +"** and select "Blueprint" from the dropdown.

4. **Connect your GitHub repository** containing this code.

5. **Configure your service**:
   - Enter a name for your service (e.g., `qwen-vl-api`)
   - Select the `Pro` plan (needed for the model size)
   - Ensure you have at least 8 GB of RAM

6. **Wait for deployment to complete** (this may take 15-30 minutes as the model is downloaded and loaded).

7. **Note your API URL** once deployment is complete.

### Configuration

The deployment can be configured using environment variables:

- `MODEL_ID`: The Hugging Face model ID (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
- `DEVICE`: Device to run the model on (default: `cpu`)
- `USE_INT8`: Whether to use 8-bit quantization (default: `true`)
- `USE_FLOAT16`: Whether to use float16 precision (default: `false`)
- `MAX_NEW_TOKENS`: Maximum number of tokens to generate (default: `1024`)

## API Usage

### Health Check

```
GET /health
```

Returns the status of the API and model information.

### Analyze an Image

```
POST /analyze
```

Request body:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that analyzes images."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What do you see in this image?"
        }
      ]
    }
  ],
  "image_data": "base64_encoded_image_data",
  "include_json": true
}
```

### Analyze a Screen with OmniParser data

```
POST /analyze_screen
```

Request body:
```json
{
  "image_data": "base64_encoded_image_data",
  "parsed_content": [
    {
      "type": "text",
      "content": "Example text",
      "box": [10, 20, 100, 50]
    }
  ],
  "task_description": "Analyze this screen for phishing elements"
}
```

## Integration with VerifiedX

To update your VerifiedX system to use this hosted API instead of running the model locally:

1. Update your `config.py` file to include the API URL:

```python
QWEN_API_URL = os.environ.get("QWEN_API_URL", "https://your-render-url.onrender.com")
```

2. Modify your QwenVLClient implementation to use the API instead of loading the model locally.

## Local Testing

To test the API locally before deployment:

```bash
pip install -r requirements.txt
python server.py
```

Then visit `http://localhost:8080` in your browser or use a tool like Postman to test the API endpoints.

## Troubleshooting

- If the model fails to load, check the logs in the Render dashboard.
- Make sure you've selected a plan with enough RAM (Pro or higher).
- Check that the disk size is large enough to store the model (at least 30GB).
- If the model is loading slowly, consider using 8-bit quantization to reduce memory usage.

## License

This project is subject to the license terms of the Qwen2.5-VL model from Hugging Face. 