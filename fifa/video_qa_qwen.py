import torch
import json
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from videoqa import BaseVQAGenerator

class Qwen25VLGenerator(BaseVQAGenerator):
    def __init__(self, model_name: str = "/data/models/huggingface-format/Qwen2.5-VL-32B-Instruct"):
        super().__init__(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def answer(self, question: str = 'Is there a dog?', video_path: str = '/home/lxj220018/Video-LLaMA/examples/dog_barking.mp4',) -> str:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            # Preparation for inference
        text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                # fps=video_kwargs["fps"],
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text[0]
