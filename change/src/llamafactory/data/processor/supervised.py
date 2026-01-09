# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        mask_history_sample: bool = False,
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        mm_input_ids, mm_labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        
        # Use encode_oneturn for mask_history_sample mode
        # All samples from mask_history_sample splitting should use encode_oneturn
        if mask_history_sample:
            # Use encode_oneturn for mask_history_sample mode
            # This handles prompts that may start with assistant messages
            prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
            
            # Apply truncation: consider mm_input_ids as part of the prompt
            total_prompt_len = len(mm_input_ids) + len(prompt_ids)
            source_len, target_len = infer_seqlen(
                total_prompt_len, len(response_ids), self.data_args.cutoff_len
            )
            
            # Truncate prompt_ids if needed
            if source_len < total_prompt_len:
                # Need to truncate prompt_ids
                prompt_ids = prompt_ids[:max(0, source_len - len(mm_input_ids))]
                source_len = len(mm_input_ids) + len(prompt_ids)
            
            # Truncate response_ids if needed
            response_ids = response_ids[:target_len]
            
            # Combine mm_plugin token ids with encoded prompt and response
            input_ids = mm_input_ids + prompt_ids + response_ids
            
            # Build labels: mask everything except the response (mask_history_sample=True)
            # Only compute loss on the last assistant response
            if self.data_args.train_on_prompt:
                labels = mm_labels + prompt_ids + response_ids
            else:
                labels = [IGNORE_INDEX] * source_len + response_ids
            
            if self.template.efficient_eos:
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_token_id]
        else:
            # Standard multiturn encoding (user-assistant alternating)
            encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
            total_length = len(mm_input_ids) + (1 if self.template.efficient_eos else 0)
            if self.data_args.mask_history:
                encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

            input_ids = mm_input_ids[:]
            labels = mm_labels[:]

            for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
                if total_length >= self.data_args.cutoff_len:
                    break

                source_len, target_len = infer_seqlen(
                    len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
                )
                source_ids = source_ids[:source_len]
                target_ids = target_ids[:target_len]
                total_length += source_len + target_len

                if self.data_args.train_on_prompt:
                    source_label = source_ids
                elif self.template.efficient_eos and turn_idx != 0:
                    source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
                else:
                    source_label = [IGNORE_INDEX] * source_len

                if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                    target_label = [IGNORE_INDEX] * target_len
                else:
                    target_label = target_ids

                if self.data_args.mask_history:  # reversed sequences
                    input_ids = source_ids + target_ids + input_ids
                    labels = source_label + target_label + labels
                else:
                    input_ids += source_ids + target_ids
                    labels += source_label + target_label

            if self.template.efficient_eos:
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]
            
            # Basic validation: response should have exactly one assistant message
            # Prompt should end with user message (can start with assistant messages for mask_history_sample)
            if len(response) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: response should have exactly one message. Got: {}".format(prompt + response)
                )
                continue
            
            # Check if this sample is from mask_history_sample splitting
            mask_history_sample = examples.get("_mask_history_sample", [False] * len(examples["_prompt"]))[i]
            
            if len(prompt) > 0:
                # Prompt should end with user message (unless it's mask_history_sample mode)
                from ..data_utils import Role
                last_prompt_role = prompt[-1].get("role")
                if not mask_history_sample and last_prompt_role != Role.USER.value:
                    logger.warning_rank0(
                        "Dropped invalid example: prompt should end with user message. Got: {}".format(prompt + response)
                    )
                    continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                mask_history_sample=mask_history_sample,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            # Ensure consistent types: convert None to empty list for media fields
            model_inputs["images"].append(examples["_images"][i] if examples["_images"][i] is not None else [])
            model_inputs["videos"].append(examples["_videos"][i] if examples["_videos"][i] is not None else [])
            model_inputs["audios"].append(examples["_audios"][i] if examples["_audios"][i] is not None else [])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # Check if this sample is from mask_history_sample splitting
            mask_history_sample = examples.get("_mask_history_sample", [False] * len(examples["_prompt"]))[i]
            
            # Validation: for standard format, prompt should have odd length (ends with user)
            # For mask_history_sample, this validation is skipped
            if not mask_history_sample:
                if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                    logger.warning_rank0(
                        "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                    )
                    continue
            
            if len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: response should have exactly one message. Got: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                mask_history_sample=mask_history_sample,
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                # Ensure consistent types: convert None to empty list for media fields
                batch_images.append(examples["_images"][i] if examples["_images"][i] is not None else [])
                batch_videos.append(examples["_videos"][i] if examples["_videos"][i] is not None else [])
                batch_audios.append(examples["_audios"][i] if examples["_audios"][i] is not None else [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
