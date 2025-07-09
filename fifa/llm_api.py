from abc import ABC, abstractmethod
class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def completion(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        给定prompt，生成模型回复
        """
        pass


import openai

class OpenaiLLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-4o", api_key: str = None, NUM_SECONDS_TO_SLEEP: int = 10):
        super().__init__(model_name)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model_name
        self.NUM_SECONDS_TO_SLEEP = NUM_SECONDS_TO_SLEEP

    def completion(self, prompt, max_tokens: int = 1024):
        model = self.model
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{
                        'role': 'system',
                        'content': 'You are a language assistant.'
                    }, {
                        'role': 'user',
                        'content': prompt,
                    }],
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p=1,
                )
                break
            # except openai.error.RateLimitError:
            #     pass
            except Exception as e:
                print(e)
            time.sleep(self.NUM_SECONDS_TO_SLEEP)
        if response.usage.total_tokens > 7000:
            print(response.usage.prompt_tokens, response.usage.total_tokens)
        return response.choices[0].message.content

from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3LLM(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        super().__init__(model_name)
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def completion(self, prompt, max_tokens: int = 1024):
        # prepare the model input
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # the result will begin with thinking content in <think></think> tags, followed by the actual response
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response







