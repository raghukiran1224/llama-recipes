# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

#RESPONSE_PROMPT = "\n\n### Response:\n{output}"
RESPONSE_PROMPT = "\n{output}"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}"
    ),
    "prompt_input_llama": (
        "<s>[INST] " \
        "<<SYS>>\nYou are a helpful, respectful and honest assistant. " \
        "Always answer as helpfully as possible, while being safe. " \
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
        "If you don't know the answer to a question, please don't share false information.\n" \
        "<</SYS>>" \
        "\n\nGenerate the next agent response by answering the question. " \
        "You are provided several documents. If the answer comes from different documents please mention all possibilities and use the titles of documents to separate between topics or domains. " \
        "If you cannot base your answer on the given documents, please state that you do not have an answer.\n\n" \
        "{input}\n\n[question]: {instruction} [/INST]"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input_llama"].format_map(ann)

        #print(f'prompt: {prompt}')
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        response = RESPONSE_PROMPT.format_map(ann)
        #print(f'response: {response}')

        response = torch.tensor(
            self.tokenizer.encode(response).append(self.tokenizer.eos_token_id), dtype=torch.int64
        )

        padding = self.max_words - (prompt.shape[0] + response.shape[0])
        if padding > 0:
            example = torch.cat((prompt, response))
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        # we truncate the prompt and always keep the response
        elif padding <= 0:
            prompt = prompt[: self.max_words - response.shape[0]]
            example = torch.cat((prompt, response))

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
