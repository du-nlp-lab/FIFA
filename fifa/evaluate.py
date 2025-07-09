


### Step 1: define your llm

## If you want to use openai
from llm_api import OpenaiLLM
llm = OpenaiLLM(model_name="gpt-4o", api_key=None, NUM_SECONDS_TO_SLEEP=10)

## If you want to use qwen
from llm_api import Qwen3LLM
llm = Qwen3LLM(model_name="Qwen/Qwen3-8B")

## If you want to use other LLMs, please rewrite BaseLLM in llm_api.py

### Step 2: define your videoqa model

### if you want to use internvl
from video_qa_internvl import InternVLGenerator
vqamodel = InternVLGenerator(model_name="/data/models/huggingface-format/InternVL2_5-8B")

### if you want to use qwen2.5-vl
from video_qa_qwen import Qwen25VLGenerator
vqamodel = Qwen25VLGenerator(model_name="/data/models/huggingface-format/Qwen2.5-VL-32B-Instruct")

## If you want to use other VideoLLMs, please rewrite BaseVQAGenerator in videoqa.py

### Step 3: Evaluate your task

## Example for Text2Video.
from eval_text2video import eval_text2video
## SHOW for Data Format
data = [{"prompt": "text", "video_path": "xx.mp4"}, {"prompt": "text", "video_path": "xx.mp4"}]
eval_text2video(data, save=False, n_parallel_workers=1, cache_dir="./results",  llm=llm, vqamodel=vqamodel)

## Example for Video2Text
from eval_video2text import eval_video2text
## SHOW for Data Format
data = [{"prompt": "text", "video_path": "xx.mp4", "question": "question"}, {"prompt": "text", "video_path": "xx.mp4", "question": "question"}]
eval_video2text(data, save=False, n_parallel_workers=1, cache_dir="./results", llm=llm, vqamodel=model)
