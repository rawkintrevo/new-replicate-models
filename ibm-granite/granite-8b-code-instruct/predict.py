# Prediction interface for Cog ⚙️
# https://cog.run/python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-8b-code-instruct")
        self.model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-8b-code-instruct", torch_dtype=torch.float16, device_map="auto")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
            system_prompt: str = Input(description="System Prompt (Optional)", default=""),
        max_new_tokens: int = Input(
            description="Maxiumum New Tokens to Generate", ge=0, le=4000, default=100
        ),
    ) -> str:
        chat = []
        if system_prompt != "":
            chat += [
                { "role": "system", "content": system_prompt },
            ]
        chat += [
            { "role": "user", "content": prompt },
        ]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # tokenize the text
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
