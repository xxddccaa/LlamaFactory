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
import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from ..extras import logging
from ..extras.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]

logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> list["MediaType"] | None:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        # Use dataset-specific media_dir if available, otherwise use global media_dir
        media_dir = self.dataset_attr.media_dir if self.dataset_attr.media_dir else self.data_args.media_dir

        if self.dataset_attr.load_from in ["script", "file"] and media_dir:
            if isinstance(medias[0], str):
                for i in range(len(medias)):
                    # Check if the media path is already an absolute path (including remote paths)
                    is_remote = medias[i].startswith(("s3://", "oss://", "s3:/", "oss:/", "gs://", "gcs://", "http://", "https://"))
                    is_absolute = os.path.isabs(medias[i]) or is_remote
                    
                    if not is_absolute:
                        # For relative paths, always use media_dir to join (even if file doesn't exist)
                        media_path = os.path.join(media_dir, medias[i])
                        medias[i] = media_path
                    # If already absolute, keep the original path
            elif isinstance(medias[0], list):  # for processed video frames
                # medias is a list of lists, e.g., [[frame1.jpg, frame2.jpg], [frame3.jpg, frame4.jpg]]
                for i in range(len(medias)):
                    for j in range(len(medias[i])):
                        # Check if the media path is already an absolute path (including remote paths)
                        is_remote = medias[i][j].startswith(("s3://", "oss://", "s3:/", "oss:/", "gs://", "gcs://", "http://", "https://"))
                        is_absolute = os.path.isabs(medias[i][j]) or is_remote
                        
                        if not is_absolute:
                            # For relative paths, always use media_dir to join (even if file doesn't exist)
                            media_path = os.path.join(media_dir, medias[i][j])
                            medias[i][j] = media_path
                        # If already absolute, keep the original path

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        # Handle mask_history_sample and max_human_steps
        if self.dataset_attr.mask_history_sample and self.dataset_attr.max_human_steps > 0:
            # Split the example into multiple samples
            outputs = self._split_example_for_mask_history(
                aligned_messages, system, example
            )
            return outputs
        else:
            output = {
                "_prompt": prompt,
                "_response": response,
                "_system": system,
                "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
                "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
                "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
                "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
            }
            return output

    def _split_example_for_mask_history(
        self, aligned_messages: list[dict[str, str]], system: str, example: dict[str, Any]
    ) -> list[dict[str, Any]]:
        r"""Split a multi-turn example into multiple samples based on max_human_steps.
        
        For example, with max_human_steps=2 and 5 assistant turns:
        Original: [user1, assistant1, user2, assistant2, user3, assistant3, user4, assistant4, user5, assistant5]
        
        Split into:
        - Sample 1: prompt=[user1], response=[assistant1]
        - Sample 2: prompt=[user1, assistant1, user2], response=[assistant2]
        - Sample 3: prompt=[assistant1, user2, assistant2, user3], response=[assistant3]
        - Sample 4: prompt=[assistant1, assistant2, user3, assistant3, user4], response=[assistant4]
        - Sample 5: prompt=[assistant1, assistant2, assistant3, user4, assistant4, user5], response=[assistant5]
        
        Each sample has at most max_human_steps human messages in the prompt, and only the last assistant
        response will be used for training (mask_history_sample=True).
        
        Media files (images, videos, audios) are also split according to the actual tokens used in each sample.
        """
        if len(aligned_messages) % 2 != 0:
            # Invalid format, return empty
            return [{
                "_prompt": [],
                "_response": [],
                "_system": system,
                "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
                "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
                "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
                "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
                "_mask_history_sample": True,  # Mark that this sample should use encode_oneturn
            }]

        num_turns = len(aligned_messages) // 2  # Number of user-assistant pairs
        max_human_steps = self.dataset_attr.max_human_steps
        outputs = []

        # Get all media files from the original example
        all_images = self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None
        all_videos = self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None
        all_audios = self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None

        # Pre-compute media token counts for each original message (in order)
        # This helps us track which media files correspond to which messages
        message_image_counts = []
        message_video_counts = []
        message_audio_counts = []
        for msg in aligned_messages:
            content = msg.get("content", "")
            message_image_counts.append(content.count(IMAGE_PLACEHOLDER))
            message_video_counts.append(content.count(VIDEO_PLACEHOLDER))
            message_audio_counts.append(content.count(AUDIO_PLACEHOLDER))

        # For each assistant turn, create a sample
        for assistant_idx in range(num_turns):
            prompt_messages = []
            prompt_message_indices = []  # Track which original messages are in prompt
            response_message_indices = []  # Track which original messages are in response
            
            # Determine the starting turn index to keep at most max_human_steps human messages
            # We want to include human messages ending at assistant_idx, with at most max_human_steps human messages
            start_turn_idx = max(0, assistant_idx - max_human_steps + 1)
            
            # Add all assistant responses before the start_turn_idx (as history context)
            for i in range(start_turn_idx):
                msg_idx = i * 2 + 1  # assistant message index
                prompt_messages.append(aligned_messages[msg_idx])
                prompt_message_indices.append(msg_idx)
            
            # Add the human messages and their corresponding assistant responses from start_turn_idx to assistant_idx
            for i in range(start_turn_idx, assistant_idx + 1):
                user_msg_idx = i * 2  # user message index
                prompt_messages.append(aligned_messages[user_msg_idx])
                prompt_message_indices.append(user_msg_idx)
                
                if i < assistant_idx:
                    assistant_msg_idx = i * 2 + 1  # assistant message index
                    prompt_messages.append(aligned_messages[assistant_msg_idx])
                    prompt_message_indices.append(assistant_msg_idx)
            
            # The response is the current assistant's message (the last one)
            response_msg_idx = assistant_idx * 2 + 1
            response_messages = [aligned_messages[response_msg_idx]]
            response_message_indices.append(response_msg_idx)

            # Count media tokens in this sample
            # We need to count the actual tokens in the messages used in this sample (prompt + response)
            # But media files are ordered sequentially from message 0 in the original conversation
            all_used_indices = sorted(set(prompt_message_indices + response_message_indices))
            if not all_used_indices:
                # No messages used, no media needed
                sample_image_count = 0
                sample_video_count = 0
                sample_audio_count = 0
            else:
                # Count actual media tokens in the messages used in this sample
                actual_image_count = sum(message_image_counts[idx] for idx in all_used_indices)
                actual_video_count = sum(message_video_counts[idx] for idx in all_used_indices)
                actual_audio_count = sum(message_audio_counts[idx] for idx in all_used_indices)
                
                # Check if prompt starts with assistant (meaning it's history context)
                # If so, we only count images from the actual messages used
                # Otherwise, we need all images from message 0 to the last message used
                if prompt_messages and prompt_messages[0]["role"] == Role.ASSISTANT.value:
                    # Prompt starts with assistant, so it's history context
                    # Only count images from the actual messages used, not from earlier messages
                    sample_image_count = actual_image_count
                    sample_video_count = actual_video_count
                    sample_audio_count = actual_audio_count
                    
                    # But we need to find the correct range in the media list
                    # Media files are ordered by message index, so we need to calculate
                    # the start index based on messages before the first used message
                    min_msg_idx = min(all_used_indices)
                    start_image_idx = sum(message_image_counts[i] for i in range(min_msg_idx))
                    start_video_idx = sum(message_video_counts[i] for i in range(min_msg_idx))
                    start_audio_idx = sum(message_audio_counts[i] for i in range(min_msg_idx))
                else:
                    # Prompt starts with user, so we need all images from message 0 to max_msg_idx
                    max_msg_idx = max(all_used_indices)
                    sample_image_count = sum(message_image_counts[i] for i in range(max_msg_idx + 1))
                    sample_video_count = sum(message_video_counts[i] for i in range(max_msg_idx + 1))
                    sample_audio_count = sum(message_audio_counts[i] for i in range(max_msg_idx + 1))
                    start_image_idx = 0
                    start_video_idx = 0
                    start_audio_idx = 0

            # Extract the corresponding media files for this sample
            # Media files are ordered by their appearance in the original conversation (from message 0)
            # Initialize as empty lists to ensure validation passes when there are no media tokens
            sample_images = []
            sample_videos = []
            sample_audios = []
            
            if sample_image_count > 0:
                if all_images is not None and len(all_images) > 0:
                    end_image_idx = start_image_idx + sample_image_count
                    if end_image_idx <= len(all_images):
                        sample_images = all_images[start_image_idx:end_image_idx]
                    else:
                        sample_images = all_images[start_image_idx:] if start_image_idx < len(all_images) else []
                        logger.warning_rank0(
                            f"Sample {assistant_idx}: Expected {sample_image_count} images starting from index {start_image_idx} "
                            f"but only {len(sample_images)} available. Total images: {len(all_images)}."
                        )
                else:
                    logger.warning_rank0(
                        f"Sample {assistant_idx}: Expected {sample_image_count} images but no images available in dataset."
                    )
            
            if sample_video_count > 0:
                if all_videos is not None and len(all_videos) > 0:
                    end_video_idx = start_video_idx + sample_video_count
                    if end_video_idx <= len(all_videos):
                        sample_videos = all_videos[start_video_idx:end_video_idx]
                    else:
                        sample_videos = all_videos[start_video_idx:] if start_video_idx < len(all_videos) else []
                        logger.warning_rank0(
                            f"Sample {assistant_idx}: Expected {sample_video_count} videos starting from index {start_video_idx} "
                            f"but only {len(sample_videos)} available. Total videos: {len(all_videos)}."
                        )
                else:
                    logger.warning_rank0(
                        f"Sample {assistant_idx}: Expected {sample_video_count} videos but no videos available in dataset."
                    )
            
            if sample_audio_count > 0:
                if all_audios is not None and len(all_audios) > 0:
                    end_audio_idx = start_audio_idx + sample_audio_count
                    if end_audio_idx <= len(all_audios):
                        sample_audios = all_audios[start_audio_idx:end_audio_idx]
                    else:
                        sample_audios = all_audios[start_audio_idx:] if start_audio_idx < len(all_audios) else []
                        logger.warning_rank0(
                            f"Sample {assistant_idx}: Expected {sample_audio_count} audios starting from index {start_audio_idx} "
                            f"but only {len(sample_audios)} available. Total audios: {len(all_audios)}."
                        )
                else:
                    logger.warning_rank0(
                        f"Sample {assistant_idx}: Expected {sample_audio_count} audios but no audios available in dataset."
                    )

            output = {
                "_prompt": prompt_messages,
                "_response": response_messages,
                "_system": system,
                "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
                "_images": sample_images,  # Keep as list (can be empty) for consistent typing in multiprocessing
                "_videos": sample_videos,  # Keep as list (can be empty) for consistent typing in multiprocessing
                "_audios": sample_audios,  # Keep as list (can be empty) for consistent typing in multiprocessing
                "_mask_history_sample": True,  # Mark that this sample should use encode_oneturn
            }
            outputs.append(output)

        return outputs


@dataclass
class OpenAIDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }

        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example.get(self.dataset_attr.system, "") if self.dataset_attr.system else ""

        aligned_messages = []
        tool_responses = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            role = message[self.dataset_attr.role_tag]
            content = message[self.dataset_attr.content_tag]

            if role in [self.dataset_attr.assistant_tag, self.dataset_attr.function_tag]:
                if "tool_calls" in message and len(message["tool_calls"]) > 0:
                    tool_calls_list = [tool["function"] for tool in message["tool_calls"]]
                    content = json.dumps(tool_calls_list, ensure_ascii=False)
                    role = self.dataset_attr.function_tag

            if role == self.dataset_attr.observation_tag:
                tool_responses.append(content)
                continue
            elif len(tool_responses) > 0:
                _content = "\n</tool_response>\n<tool_response>\n".join(tool_responses)
                aligned_messages.append(
                    {
                        "role": Role.OBSERVATION.value,
                        "content": _content,
                    }
                )
                tool_responses = []

            aligned_messages.append(
                {
                    "role": tag_mapping[role],
                    "content": content,
                }
            )

        odd_tags = (Role.USER.value, Role.OBSERVATION.value)
        even_tags = (Role.ASSISTANT.value, Role.FUNCTION.value)
        accept_tags = (odd_tags, even_tags)
        for turn_idx, message in enumerate(aligned_messages):
            if message["role"] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        tools = example.get(self.dataset_attr.tools, "") if self.dataset_attr.tools else ""
        if isinstance(tools, dict) or isinstance(tools, list):
            tools = json.dumps(tools, ensure_ascii=False)

        short_system_prompt = "detailed thinking off"
        if not system:
            if not tools:
                system = short_system_prompt
            else:
                pass
        else:
            if not tools:
                if "detailed thinking on" in system or "detailed thinking off" in system:
                    pass
                else:
                    system += "\n" + short_system_prompt
            else:
                system += "\n"

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": tools,
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "openai": OpenAIDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _tools: "..."
    _images: []
    _videos: []
    _audios: []
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    
    # Check if converter might return lists (mask_history_sample is enabled)
    if dataset_attr.mask_history_sample and dataset_attr.max_human_steps > 0:
        # Process first example for debugging/logging
        first_example = next(iter(dataset))
        first_example_dict = {key: first_example[key] for key in column_names}
        first_result = dataset_converter(first_example_dict)
        
        logger.info_rank0("=" * 80)
        logger.info_rank0("Mask History Sample Debug: First Example Processing")
        logger.info_rank0("=" * 80)
        logger.info_rank0(f"Original example (first {min(3, len(column_names))} keys shown):")
        for key in list(column_names)[:3]:
            value = first_example_dict.get(key)
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    logger.info_rank0(f"  {key}: [{len(value)} items]")
                    if key == dataset_attr.messages:
                        logger.info_rank0(f"    Messages count: {len(value)}")
                        for idx, msg in enumerate(value[:5]):  # Show first 5 messages
                            role = msg.get(dataset_attr.role_tag, "unknown")
                            content = msg.get(dataset_attr.content_tag, "")[:100]
                            image_count = content.count(IMAGE_PLACEHOLDER)
                            logger.info_rank0(f"      [{idx}] {role}: {image_count} <image> tokens, content: {content}...")
                else:
                    logger.info_rank0(f"  {key}: {value[:3] if len(value) > 3 else value}...")
            elif isinstance(value, str):
                logger.info_rank0(f"  {key}: {value[:100]}...")
            else:
                logger.info_rank0(f"  {key}: {value}")
        
        if dataset_attr.images and first_example_dict.get(dataset_attr.images):
            images = first_example_dict[dataset_attr.images]
            logger.info_rank0(f"  {dataset_attr.images}: {len(images)} images")
            if isinstance(images, list) and len(images) > 0:
                logger.info_rank0(f"    First 3: {images[:3]}")
        
        logger.info_rank0(f"\nSplit result: {len(first_result) if isinstance(first_result, list) else 1} sample(s)")
        if isinstance(first_result, list):
            for idx, sample in enumerate(first_result):
                prompt = sample.get("_prompt", [])
                response = sample.get("_response", [])
                images = sample.get("_images", [])
                
                # Count user and assistant in prompt
                user_count = sum(1 for msg in prompt if msg.get("role") == "user")
                assistant_count = sum(1 for msg in prompt if msg.get("role") == "assistant")
                
                logger.info_rank0(f"\n  Sample {idx + 1}:")
                logger.info_rank0(f"    Prompt: {len(prompt)} messages (user: {user_count}, assistant: {assistant_count})")
                logger.info_rank0(f"    Response: {len(response)} messages")
                logger.info_rank0(f"    Images: {len(images) if images else 0}")
                
                # Count image tokens in prompt + response
                total_image_tokens = 0
                for msg in prompt + response:
                    total_image_tokens += msg.get("content", "").count(IMAGE_PLACEHOLDER)
                logger.info_rank0(f"    <image> tokens: {total_image_tokens}")
                
                # Show prompt structure
                logger.info_rank0(f"    Prompt structure:")
                for msg_idx, msg in enumerate(prompt):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:60]
                    # Mark history assistants (those before first user)
                    if role == "assistant" and msg_idx < len(prompt) and all(m.get("role") != "user" for m in prompt[:msg_idx+1]):
                        role_label = "assistant[history]"
                    else:
                        role_label = role
                    logger.info_rank0(f"      [{msg_idx}] {role_label:20s}: {content}...")
                
                # Show response
                if len(response) > 0:
                    role = response[0].get("role", "unknown")
                    content = response[0].get("content", "")[:60]
                    logger.info_rank0(f"    Response:")
                    logger.info_rank0(f"      [0] {role:20s}: {content}...")
        logger.info_rank0("=" * 80)
        
        # Use batched mode to properly handle list returns
        def batched_converter_wrapper(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            """Process examples in batch, handling list returns from converter."""
            outputs = []
            batch_size = len(examples[column_names[0]])
            for i in range(batch_size):
                example = {key: examples[key][i] for key in column_names}
                result = dataset_converter(example)
                if isinstance(result, list):
                    outputs.extend(result)
                else:
                    outputs.append(result)
            
            # Convert list of dicts to dict of lists
            if not outputs:
                # Return empty structure with all expected keys
                # This handles the case where all examples are invalid
                return {
                    "_prompt": [],
                    "_response": [],
                    "_system": [],
                    "_tools": [],
                    "_images": [],
                    "_videos": [],
                    "_audios": [],
                    "_mask_history_sample": [],
                }
            
            # Get all unique keys from all outputs
            all_keys = set()
            for output in outputs:
                all_keys.update(output.keys())
            
            # Normalize outputs to ensure consistent types for PyArrow
            # Convert None to empty list/string to avoid type casting errors in multiprocessing
            for output in outputs:
                if "_images" in output and output["_images"] is None:
                    output["_images"] = []
                if "_videos" in output and output["_videos"] is None:
                    output["_videos"] = []
                if "_audios" in output and output["_audios"] is None:
                    output["_audios"] = []
                if "_system" in output and output["_system"] is None:
                    output["_system"] = ""
                if "_tools" in output and output["_tools"] is None:
                    output["_tools"] = ""
            
            result_dict = {}
            for key in all_keys:
                if key == "_mask_history_sample":
                    # Ensure _mask_history_sample has default value False instead of None
                    result_dict[key] = [output.get(key, False) for output in outputs]
                elif key == "_images" or key == "_videos" or key == "_audios":
                    # Ensure media fields always have list type (empty list instead of None)
                    result_dict[key] = [output.get(key) if output.get(key) is not None else [] for output in outputs]
                elif key == "_system" or key == "_tools":
                    # Ensure string fields always have string type (empty string instead of None)
                    result_dict[key] = [output.get(key) if output.get(key) is not None else "" for output in outputs]
                else:
                    result_dict[key] = [output.get(key) for output in outputs]
            return result_dict
        
        # Use batched mode to properly handle list returns
        # Process one batch at a time (batch_size=1) to avoid mixing different number of split samples
        # This ensures PyArrow can properly infer schema without type conflicts
        aligned_dataset = dataset.map(
            batched_converter_wrapper,
            batched=True,
            batch_size=1,  # Process one example at a time to handle list returns
            remove_columns=column_names,
            **kwargs,
        )
        
        # Log statistics after processing
        # Count original examples by checking the dataset before mapping
        original_count = len(dataset) if hasattr(dataset, '__len__') else None
        final_count = len(aligned_dataset) if hasattr(aligned_dataset, '__len__') else None
        
        logger.info_rank0("=" * 80)
        logger.info_rank0("Mask History Sample Statistics")
        logger.info_rank0("=" * 80)
        if original_count is not None:
            logger.info_rank0(f"Original examples: {original_count}")
        if final_count is not None:
            logger.info_rank0(f"Final dataset size (after splitting): {final_count}")
            if original_count is not None and original_count > 0:
                logger.info_rank0(f"Average samples per example: {final_count / original_count:.2f}")
        logger.info_rank0("=" * 80)
        
        # Apply type normalization to the entire dataset to ensure consistency in multiprocessing
        # This step ensures all None values are converted to appropriate empty values
        def normalize_types(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            """Normalize types to ensure consistency for PyArrow in multiprocessing."""
            for key in examples.keys():
                if key in ["_images", "_videos", "_audios"]:
                    # Ensure media fields are always lists (convert None to empty list)
                    examples[key] = [v if v is not None else [] for v in examples[key]]
                elif key in ["_system", "_tools"]:
                    # Ensure string fields are always strings (convert None to empty string)
                    examples[key] = [v if v is not None else "" for v in examples[key]]
                elif key == "_mask_history_sample":
                    # Ensure boolean field has default value False
                    examples[key] = [v if v is not None else False for v in examples[key]]
            return examples
        
        # Apply normalization without changing the number of rows
        normalize_kwargs = {}
        if not data_args.streaming:
            normalize_kwargs = dict(
                num_proc=1,  # Use single process to avoid schema conflicts during normalization
                load_from_cache_file=False,
                desc="Normalizing types for multiprocessing",
            )
        
        aligned_dataset = aligned_dataset.map(
            normalize_types,
            batched=True,
            batch_size=100,  # Process in larger batches for efficiency
            **normalize_kwargs,
        )
        
        # Note: We skip explicit schema casting because PyArrow has difficulty casting
        # complex nested structures (list of structs). The normalization step above
        # ensures type consistency by converting None to appropriate empty values,
        # which is sufficient for multiprocessing compatibility.
        # The downstream tokenizer processing will use single process (num_proc=1)
        # for mask_history_sample datasets to avoid schema conflicts.
        
        return aligned_dataset
    else:
        # Normal mode: converter returns single dict
        return dataset.map(
            dataset_converter,
            batched=False,
            remove_columns=column_names,
            **kwargs,
        )
