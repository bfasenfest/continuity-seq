# ContinuitySeq

Variable length frame conditioning for infinite length video generation. Can be used to generate videos from text that continue from an existing video or image. It can also generate the initial image using a standard text-to-image model from huggingface. Developed in collaboration with [motexture](https://github.com/motexture) and based on [vseq2vseq](https://github.com/motexture/vseq2vseq).

Inital checkpoint can be found at [b-f/ContinuitySeq-A](https://huggingface.co/b-f/ContinuitySeq-A).

## Running Inference

Using a custom 2d text to image diffusion model for image conditioning:

```python
inference.py \
  --model b-f/ContinuitySeq-A \
  --prompt "an astronaut is walking on the moon" \
  --model-2d stabilityai/stable-diffusion-2-1 \
  --num-frames 16 \
  --width 576 \
  --height 320 \
  --times 1 \
  --sdp
```

Animating a static image:

```python
inference.py \
  --model b-f/ContinuitySeq-A \
  --prompt "an astronaut is walking on the moon" \
  --init-image "image.png" \
  --num-frames 16 \
  --times 1 \
  --sdp
```

Creating infinite length videos by using the last frame as the new init image and by increasing the --times parameter:

```python
inference.py \
  --model b-f/ContinuitySeq-A \
  --prompt "an astronaut is walking on the moon" \
  --model-2d stabilityai/stable-diffusion-2-1 \
  --num-frames 16 \
  --width 576 \
  --height 320 \
  --times 4 \
  --sdp
```

```bash
options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        HuggingFace repository or path to model checkpoint directory
  -p PROMPT, --prompt PROMPT
                        Text prompt to condition on
  -t TIMES, --times TIMES
                        How many times to continue to generate videos
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save output video to
  -n NEGATIVE_PROMPT, --negative-prompt NEGATIVE_PROMPT
                        Text prompt to condition against
  -T NUM_FRAMES, --num-frames NUM_FRAMES
                        Total number of frames to generate
  -CN MIN_CONDITIONING_N_SAMPLE_FRAMES, --min_conditioning_n_sample_frames MIN_CONDITIONING_N_SAMPLE_FRAMES
                        Total number of frames to sample for conditioning after initial video
  -CX MAX_CONDITIONING_N_SAMPLE_FRAMES, --max_conditioning_n_sample_frames MAX_CONDITIONING_N_SAMPLE_FRAMES
                        Total number of frames to sample for conditioning after initial video
  -WI WIDTH, --width WIDTH
                        Width of the video to generate (if init image is not provided)
  -HI HEIGHT, --height HEIGHT
                        Height of the video (if init image is not provided)
  -IW IMAGE_WIDTH, --image-width IMAGE_WIDTH
                        Width of the image to generate (if init image is not provided)
  -IH IMAGE_HEIGHT, --image-height IMAGE_HEIGHT
                        Height of the image (if init image is not provided)
  -MP MODEL_2D, --model-2d MODEL_2D
                        Path to the model for image generation (if init image is not provided)
  -i INIT_IMAGE, --init-image INIT_IMAGE
                        Path to initial image to use
  -VS VAE_SCALE, --vae-scale VAE_SCALE
                        VAE scale factor
  -VB VAE_BATCH_SIZE, --vae-batch-size VAE_BATCH_SIZE
                        Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).
  -s NUM_STEPS, --num-steps NUM_STEPS
                        Number of diffusion steps to run per frame.
  -g GUIDANCE_SCALE, --guidance-scale GUIDANCE_SCALE
                        Scale for guidance loss (higher values = more guidance, but possibly more artifacts).
  -IG IMAGE_GUIDANCE_SCALE, --image-guidance-scale IMAGE_GUIDANCE_SCALE
                        Scale for guidance loss for 2d model (higher values = more guidance, but possibly more artifacts).
  -f FPS, --fps FPS     FPS of output video
  -d DEVICE, --device DEVICE
                        Device to run inference on (defaults to cuda).
  -x, --xformers        Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).
  -S, --sdp             Use SDP attention, PyTorch's built-in memory-efficient attention implementation.
  -r SEED, --seed SEED  Random seed to make generations reproducible.
  -I, --save-init       Save the init image to the output folder for reference
  -N, --include-model   Include the name of the model in the exported file
  -u, --upscale         Use a latent upscaler
```

## Shoutouts

- [motexture](https://github.com/motexture) for developing the original model and training code
- [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning/) for work on modelscope training and inference code
- [cerspense](https://github.com/cerspense) for help with training parameters and settings
