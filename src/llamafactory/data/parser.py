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
from dataclasses import dataclass
from typing import Any, Literal

from huggingface_hub import hf_hub_download

from ..extras.constants import DATA_CONFIG, FILEEXT2TYPE
from ..extras.misc import use_modelscope, use_openmind


@dataclass
class DatasetAttr:
    r"""Dataset attributes."""

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt", "openai"] = "alpaca"
    ranking: bool = False
    # extra configs
    subset: str | None = None
    split: str = "train"
    folder: str | None = None
    num_samples: int | None = None
    media_dir: str | None = None  # Media directory for this dataset
    # common columns
    system: str | None = None
    tools: str | None = None
    images: str | None = None
    videos: str | None = None
    audios: str | None = None
    # dpo columns
    chosen: str | None = None
    rejected: str | None = None
    kto_tag: str | None = None
    # alpaca columns
    prompt: str | None = "instruction"
    query: str | None = "input"
    response: str | None = "output"
    history: str | None = None
    # sharegpt columns
    messages: str | None = "conversations"
    # sharegpt tags
    role_tag: str | None = "from"
    content_tag: str | None = "value"
    user_tag: str | None = "human"
    assistant_tag: str | None = "gpt"
    observation_tag: str | None = "observation"
    function_tag: str | None = "function_call"
    system_tag: str | None = "system"
    # sample processing configs
    mask_history_sample: bool = False
    max_human_steps: int = -1

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Any | None = None) -> None:
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        self.set_attr("formatting", attr, default="alpaca")
        self.set_attr("ranking", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("folder", attr)
        self.set_attr("num_samples", attr)
        self.set_attr("media_dir", attr)

        if "columns" in attr:
            column_names = ["prompt", "query", "response", "history", "messages", "system", "tools"]
            column_names += ["images", "videos", "audios", "chosen", "rejected", "kto_tag"]
            for column_name in column_names:
                self.set_attr(column_name, attr["columns"])

        if "tags" in attr:
            tag_names = ["role_tag", "content_tag"]
            tag_names += ["user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]
            for tag in tag_names:
                self.set_attr(tag, attr["tags"])

        # Set sample processing configs
        self.set_attr("mask_history_sample", attr, default=False)
        self.set_attr("max_human_steps", attr, default=-1)

        # Validate mask_history_sample and max_human_steps
        # They must be set together (both default or both non-default)
        has_mask_history = self.mask_history_sample != False
        has_max_steps = self.max_human_steps != -1

        if has_mask_history != has_max_steps:
            raise ValueError(
                f"mask_history_sample and max_human_steps must be set together. "
                f"Got mask_history_sample={self.mask_history_sample}, max_human_steps={self.max_human_steps}. "
                f"Either both should use default values (mask_history_sample=False, max_human_steps=-1) "
                f"or both should be set explicitly."
            )

        if has_max_steps and self.max_human_steps < 1:
            raise ValueError(
                f"max_human_steps must be >= 1 when set. Got max_human_steps={self.max_human_steps}."
            )


def _parse_dataset_path_with_config(path_with_config: str) -> tuple[str, dict[str, Any]]:
    r"""Parse dataset path with optional configuration in brackets.
    
    Supports syntax like:
    - /mnt/data.json[media_dir=/mnt/images]
    - /mnt/data.json[media_dir=/mnt/images,formatting=alpaca]
    - s3://bucket/data.json[media_dir=s3://bucket/images,user_tag=user]
    
    Returns:
        tuple: (file_path, config_dict)
    """
    # Check if there's a config section in brackets
    if '[' not in path_with_config:
        return path_with_config, {}
    
    # Find the last '[' to handle paths that might contain '['
    bracket_start = path_with_config.rfind('[')
    if bracket_start == -1 or not path_with_config.endswith(']'):
        return path_with_config, {}
    
    file_path = path_with_config[:bracket_start]
    config_str = path_with_config[bracket_start+1:-1]  # Remove [ and ]
    
    if not config_str.strip():
        return file_path, {}
    
    # Parse config string: "key1=value1,key2=value2"
    config_dict = {}
    # Split by comma, but be careful with values that might contain commas in paths
    parts = []
    current_part = []
    in_value = False
    
    for char in config_str:
        if char == '=' and not in_value:
            in_value = True
            current_part.append(char)
        elif char == ',' and in_value:
            # Check if this is likely a path separator (followed by space or another key)
            # For simplicity, we assume ',' separates key-value pairs
            parts.append(''.join(current_part))
            current_part = []
            in_value = False
        else:
            current_part.append(char)
    
    if current_part:
        parts.append(''.join(current_part))
    
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Convert boolean strings
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        # Convert numeric strings
        elif value.isdigit():
            value = int(value)
        
        config_dict[key] = value
    
    return file_path, config_dict


def _is_file_path(path: str) -> bool:
    r"""Check if the path is a file path (local or remote).
    
    Note: This function should be called after _parse_dataset_path_with_config
    to remove any bracketed configuration.
    """
    # Check for remote paths (s3://, oss://, gs://, gcs://, etc.)
    if path.startswith(("s3://", "oss://", "s3:/", "oss:/", "gs://", "gcs://")):
        return True
    
    # Check if it looks like an absolute path (starts with /)
    # This is important for paths that might not exist yet
    if os.path.isabs(path):
        return True
    
    # Check for local file paths that exist
    if os.path.isfile(path) or os.path.isdir(path):
        return True
    
    # Check if it has a file extension, which suggests it's a file path
    # Common data file extensions
    file_extensions = ('.json', '.jsonl', '.csv', '.parquet', '.arrow', '.txt')
    if any(path.lower().endswith(ext) for ext in file_extensions):
        # Further check: if it contains path separators, likely a file path
        if '/' in path or '\\' in path:
            return True
    
    return False


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str | dict) -> list["DatasetAttr"]:
    r"""Get the attributes of the datasets."""
    if dataset_names is None:
        dataset_names = []

    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    elif dataset_dir == "ONLINE":
        dataset_info = None
    else:
        if dataset_dir.startswith("REMOTE:"):
            config_path = hf_hub_download(repo_id=dataset_dir[7:], filename=DATA_CONFIG, repo_type="dataset")
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG)

        try:
            with open(config_path) as f:
                dataset_info = json.load(f)
        except Exception as err:
            # If dataset_info.json doesn't exist, we'll still try to handle file paths directly
            # Only raise error if there are dataset_names that are not file paths
            if len(dataset_names) != 0:
                # Check if any dataset_names are not file paths (need dataset_info.json)
                # Parse paths first to remove inline config before checking
                has_non_file_paths = any(
                    not _is_file_path(_parse_dataset_path_with_config(name)[0]) 
                    for name in dataset_names
                )
                if has_non_file_paths:
                    raise ValueError(f"Cannot open {config_path} due to {str(err)}.")
                # All are file paths, can handle directly
                dataset_info = None
            else:
                dataset_info = None

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        # Parse dataset path with optional configuration
        # Supports syntax like: /mnt/data.json[media_dir=/mnt/images,formatting=alpaca]
        file_path, inline_config = _parse_dataset_path_with_config(name)
        
        # First check if name is a file path (allowing mixed usage)
        if _is_file_path(file_path):
            # Direct file path: use file loader with default sharegpt format
            dataset_attr = DatasetAttr(
                load_from="file",
                dataset_name=file_path,
                formatting=inline_config.get("formatting", "sharegpt"),
                messages=inline_config.get("messages", "conversations"),
                role_tag=inline_config.get("role_tag", "from"),
                content_tag=inline_config.get("content_tag", "value"),
                user_tag=inline_config.get("user_tag", "human"),
                assistant_tag=inline_config.get("assistant_tag", "gpt"),
            )
            
            # Apply additional inline configurations
            if "media_dir" in inline_config:
                dataset_attr.media_dir = inline_config["media_dir"]
            if "images" in inline_config:
                dataset_attr.images = inline_config["images"]
            if "videos" in inline_config:
                dataset_attr.videos = inline_config["videos"]
            if "audios" in inline_config:
                dataset_attr.audios = inline_config["audios"]
            if "system" in inline_config:
                dataset_attr.system = inline_config["system"]
            if "tools" in inline_config:
                dataset_attr.tools = inline_config["tools"]
            if "mask_history_sample" in inline_config:
                dataset_attr.mask_history_sample = inline_config["mask_history_sample"]
            if "max_human_steps" in inline_config:
                dataset_attr.max_human_steps = inline_config["max_human_steps"]
            
            # Validate mask_history_sample and max_human_steps
            has_mask_history = dataset_attr.mask_history_sample != False
            has_max_steps = dataset_attr.max_human_steps != -1
            if has_mask_history != has_max_steps:
                raise ValueError(
                    f"For dataset {file_path}: mask_history_sample and max_human_steps must be set together. "
                    f"Got mask_history_sample={dataset_attr.mask_history_sample}, max_human_steps={dataset_attr.max_human_steps}. "
                    f"Either both should use default values (mask_history_sample=False, max_human_steps=-1) "
                    f"or both should be set explicitly."
                )
            if has_max_steps and dataset_attr.max_human_steps < 1:
                raise ValueError(
                    f"For dataset {file_path}: max_human_steps must be >= 1 when set. "
                    f"Got max_human_steps={dataset_attr.max_human_steps}."
                )
            
            dataset_list.append(dataset_attr)
            continue

        # Not a file path, check dataset_info.json
        if dataset_info is None:  # dataset_dir is ONLINE or dataset_info.json not found
            # Not a file path and no dataset_info.json, treat as hub dataset
            load_from = "ms_hub" if use_modelscope() else "om_hub" if use_openmind() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        # Try to find in dataset_info.json
        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        elif "cloud_file_name" in dataset_info[name]:
            dataset_attr = DatasetAttr("cloud_file", dataset_name=dataset_info[name]["cloud_file_name"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.join(dataset_info[name])
        
        # Resolve media_dir if it's a relative path: make it relative to dataset_dir
        # Similar to how file_name is resolved in loader.py
        if dataset_attr.media_dir and not isinstance(dataset_dir, dict) and dataset_dir != "ONLINE":
            # Check if media_dir is a remote path or absolute path
            is_remote = dataset_attr.media_dir.startswith(("s3://", "oss://", "s3:/", "oss:/", "gs://", "gcs://", "http://", "https://"))
            is_absolute = os.path.isabs(dataset_attr.media_dir) or is_remote
            
            if not is_absolute:
                # For relative paths, resolve relative to dataset_dir
                dataset_attr.media_dir = os.path.join(dataset_dir, dataset_attr.media_dir)
        
        dataset_list.append(dataset_attr)

    return dataset_list
