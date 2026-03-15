export VLLM_DEBUG_VIDEO_DUMP_DIR="${VLLM_DEBUG_VIDEO_DUMP_DIR:-/tmp/vllm_video_dumps}"

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-4B \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --allowed-local-media-path /dss/mcmlscratch/03/di35dov

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching

vllm serve Qwen/Qwen3.5-27B-FP8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
