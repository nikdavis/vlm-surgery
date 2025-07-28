# Use NVIDIA vLLM image with Triton
FROM nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3

WORKDIR /app

# Just copy the minimal server file
COPY vllm_server.py .

# Use vLLM's built-in server
CMD ["python", "vllm_server.py"]