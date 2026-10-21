import os
import sys
import time
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.vision.qwen_service import Qwen2VLService, ImagePreprocessor, VisualContextParser

def profile_qwen():
    image_path = "samples_LEVIR/A/test_2_0000_0000.png"
    if not os.path.exists(image_path):
        print(f"Error: image not found {image_path}")
        return

    print("=== PROFILING QWEN2-VL IMPLEMENTATION ===")
    service = Qwen2VLService.get_instance()
    
    t0 = time.time()
    service.load_model()
    load_time = time.time() - t0
    print(f"Model Load Time: {load_time:.2f}s")
    print(f"Device: {service._device}")
    if service._device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
        print(f"Reserved VRAM: {torch.cuda.memory_reserved() / (1024**2):.1f} MB")

    # 1. Preprocessing
    t_p0 = time.time()
    pil_img = ImagePreprocessor.preprocess(image_path)
    prep_time = time.time() - t_p0
    print(f"Preprocessed Image Size: {pil_img.size}, Mode: {pil_img.mode}, Time: {prep_time*1000:.2f}ms")

    # 2. Processor tokenization
    t_tok0 = time.time()
    from satquery.vision.qwen_service import QWEN_VISION_PROMPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": QWEN_VISION_PROMPT}
            ]
        }
    ]
    text_prompt = service._processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = service._processor(
        text=[text_prompt],
        images=[pil_img],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(service._device)
    tok_time = time.time() - t_tok0
    
    input_ids_shape = inputs.input_ids.shape
    image_grid_thw = inputs.image_grid_thw if hasattr(inputs, "image_grid_thw") else None
    print(f"Processor Tokenization Time: {tok_time*1000:.2f}ms")
    print(f"Input IDs shape: {input_ids_shape} (Total input tokens: {input_ids_shape[1]})")
    print(f"Image Grid THW: {image_grid_thw}")

    # 3. Model Generation
    if service._device.type == "cuda":
        torch.cuda.synchronize()
    t_gen0 = time.time()
    with torch.inference_mode():
        generated_ids = service._model.generate(
            **inputs,
            max_new_tokens=service.max_new_tokens,
            do_sample=False,
            use_cache=True
        )
    if service._device.type == "cuda":
        torch.cuda.synchronize()
    gen_time = time.time() - t_gen0
    
    num_generated_tokens = generated_ids.shape[1] - input_ids_shape[1]
    print(f"Generation Time: {gen_time:.2f}s")
    print(f"Generated Tokens: {num_generated_tokens} tokens ({num_generated_tokens/gen_time:.2f} tokens/sec)")

    # 4. Decoding and Parsing
    t_dec0 = time.time()
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = service._processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed = VisualContextParser.parse(output_text)
    dec_time = time.time() - t_dec0
    print(f"Decode and Parse Time: {dec_time*1000:.2f}ms")
    print(f"Total Vision Time: {prep_time + tok_time + gen_time + dec_time:.2f}s")
    print(f"Parsed Fields Count: {len(parsed)}")

if __name__ == "__main__":
    profile_qwen()
