import os
import random
import json

import re

import torch
import sys
sys.path.append('../')

class TextPromptDataset:
    def __init__(self, path, num=None):
        super().__init__()

        jsonl_file_path = path
        prompt_path = re.sub(r'(train|valid)_anno', r'\1_prompt', path)

        self.prompts = []
        if not os.path.exists(prompt_path):
            with open(jsonl_file_path, 'r') as jsonl_file:
                for line in jsonl_file:
                    json_object = json.loads(line)
                    key = list(json_object['Task2'].keys())[0]
                    assert key in ['Caption', 'Caption:', 'caption']
                    self.prompts += [json_object['Task2'][key]]
            self.prompts = list(set(filter(None, self.prompts)))
            self.prompts.sort()
            random.seed(2023)
            random.shuffle(self.prompts)
            with open(prompt_path, 'w') as jsonl_file:
                json.dump(self.prompts, jsonl_file)
        else:
            with open(prompt_path, 'r') as jsonl_file:
                self.prompts = json.load(jsonl_file)

        # self.prompts = self.prompts \
        #     if num is None else self.prompts[:num]

    def shuffle(self, *args, **kwargs):
        random.shuffle(self.prompts)
        return self

    def select(self, selected_range):
        self.prompts = [self.prompts[idx] for idx in selected_range]
        return self

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                self.prompts[index],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
            )[0]
        data = {"prompt_embeds": prompt_embeds.squeeze(0), "prompt": self.prompts[index]}
        return data