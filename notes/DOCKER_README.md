# Pixtral OCR Inference Docker Container

This Docker container provides a FastAPI server for running inference with the fine-tuned Pixtral OCR model.

## Features

- FastAPI REST API with automatic documentation
- Base64 image input or file upload
- Support for both XML and JSON output formats
- Configurable temperature and max tokens
- GPU acceleration with 4-bit quantization
- Health check endpoint

## Building and Running

### Build the container:
```bash
cd docker
docker-compose build
```

### Run the container:
```bash
docker-compose up
```

The server will be available at `http://localhost:8000`

## API Endpoints

### `/inference` (POST)
Main inference endpoint with base64 encoded image.

Request body:
```json
{
  "image_base64": "...",
  "ocr_text": "...",
  "output_format": "xml",  // or "json"
  "prompt": "...",         // optional custom prompt
  "temperature": 0.4,      // optional
  "max_tokens": 4096       // optional
}
```

### `/inference/file` (POST)
File upload endpoint for inference.

Form data:
- `image`: Image file
- `ocr_text`: OCR text string
- `output_format`: "xml" or "json"
- `prompt`: Optional custom prompt
- `temperature`: Optional (default 0.4)
- `max_tokens`: Optional (default 4096)

### `/health` (GET)
Health check endpoint.

### `/docs` (GET)
Interactive API documentation (Swagger UI).

## Testing

Use the provided test client:

```bash
# Test with base64 encoding
python test_client.py \
  --image ../ready/enc_brit_v15_p0470/enc_brit_v15_p0470.jpg \
  --ocr ../ready/enc_brit_v15_p0470/enc_brit_v15_p0470_ocr.txt \
  --format xml

# Test with file upload
python test_client.py \
  --image ../ready/enc_brit_v15_p0470/enc_brit_v15_p0470.jpg \
  --ocr ../ready/enc_brit_v15_p0470/enc_brit_v15_p0470_ocr.txt \
  --format json \
  --upload

# Test with custom prompt
python test_client.py \
  --image ../ready/enc_brit_v15_p0470/enc_brit_v15_p0470.jpg \
  --ocr ../ready/enc_brit_v15_p0470/enc_brit_v15_p0470_ocr.txt \
  --prompt "Extract as XML with header and articles"
```

## Environment Variables

Configure via docker-compose.yml or environment:

- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 2)
- `MODEL_PATH`: Path to model directory (default: /app/model)
- `LOAD_IN_4BIT`: Enable 4-bit quantization (default: true)
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 4096)
- `TEMPERATURE`: Sampling temperature (default: 0.4)

## Performance

- Model loads on startup (~30 seconds)
- Inference time depends on image size and output length
- Typical response time: 5-30 seconds per request
- Memory usage: ~15GB GPU memory with 4-bit quantization

## Troubleshooting

1. **CUDA out of memory**: Adjust `CUDA_VISIBLE_DEVICES` to use a different GPU
2. **Model not loading**: Check that the model path is correctly mounted
3. **Slow inference**: Try reducing `MAX_NEW_TOKENS` or image size
4. **Poor quality output**: Increase temperature or continue training the model