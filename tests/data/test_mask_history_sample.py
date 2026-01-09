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

import pytest

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


def test_mask_history_sample_converter_split():
    """Test that converter splits examples correctly with mask_history_sample and max_human_steps."""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.mask_history_sample = True
    dataset_attr.max_human_steps = 2

    data_args = DataArguments()

    # Create a multi-turn example with 5 assistant turns
    example = {
        "conversations": [
            {"from": "human", "value": "user1"},
            {"from": "gpt", "value": "assistant1"},
            {"from": "human", "value": "user2"},
            {"from": "gpt", "value": "assistant2"},
            {"from": "human", "value": "user3"},
            {"from": "gpt", "value": "assistant3"},
            {"from": "human", "value": "user4"},
            {"from": "gpt", "value": "assistant4"},
            {"from": "human", "value": "user5"},
            {"from": "gpt", "value": "assistant5"},
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    outputs = dataset_converter(example)

    # Should return a list of 5 samples (one for each assistant turn)
    assert isinstance(outputs, list)
    assert len(outputs) == 5

    # Check first sample: [user1] -> [assistant1]
    assert outputs[0]["_prompt"] == [{"role": Role.USER.value, "content": "user1"}]
    assert outputs[0]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant1"}]
    assert outputs[0]["_mask_history_sample"] is True

    # Check second sample: [user1, assistant1, user2] -> [assistant2]
    assert len(outputs[1]["_prompt"]) == 3
    assert outputs[1]["_prompt"][0]["role"] == Role.USER.value
    assert outputs[1]["_prompt"][1]["role"] == Role.ASSISTANT.value
    assert outputs[1]["_prompt"][2]["role"] == Role.USER.value
    assert outputs[1]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant2"}]
    assert outputs[1]["_mask_history_sample"] is True

    # Check third sample: [assistant1, user2, assistant2, user3] -> [assistant3]
    # Should start with assistant (history context)
    assert len(outputs[2]["_prompt"]) == 4
    assert outputs[2]["_prompt"][0]["role"] == Role.ASSISTANT.value
    assert outputs[2]["_prompt"][1]["role"] == Role.USER.value
    assert outputs[2]["_prompt"][2]["role"] == Role.ASSISTANT.value
    assert outputs[2]["_prompt"][3]["role"] == Role.USER.value
    assert outputs[2]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant3"}]
    assert outputs[2]["_mask_history_sample"] is True

    # Check that all samples have the mask_history_sample flag
    for output in outputs:
        assert output["_mask_history_sample"] is True
        assert len(output["_response"]) == 1
        assert output["_response"][0]["role"] == Role.ASSISTANT.value


def test_mask_history_sample_max_human_steps():
    """Test that max_human_steps limits the number of human messages in prompt."""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.mask_history_sample = True
    dataset_attr.max_human_steps = 2

    data_args = DataArguments()

    # Create example with 5 turns
    example = {
        "conversations": [
            {"from": "human", "value": "user1"},
            {"from": "gpt", "value": "assistant1"},
            {"from": "human", "value": "user2"},
            {"from": "gpt", "value": "assistant2"},
            {"from": "human", "value": "user3"},
            {"from": "gpt", "value": "assistant3"},
            {"from": "human", "value": "user4"},
            {"from": "gpt", "value": "assistant4"},
            {"from": "human", "value": "user5"},
            {"from": "gpt", "value": "assistant5"},
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    outputs = dataset_converter(example)

    # Check that each sample has at most max_human_steps (2) user messages in prompt
    for i, output in enumerate(outputs):
        prompt = output["_prompt"]
        user_count = sum(1 for msg in prompt if msg["role"] == Role.USER.value)
        assert user_count <= 2, f"Sample {i} has {user_count} user messages, should be <= 2"


def test_mask_history_sample_without_flag():
    """Test that normal examples without mask_history_sample work as before."""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.mask_history_sample = False
    dataset_attr.max_human_steps = -1

    data_args = DataArguments()

    example = {
        "conversations": [
            {"from": "human", "value": "user1"},
            {"from": "gpt", "value": "assistant1"},
            {"from": "human", "value": "user2"},
            {"from": "gpt", "value": "assistant2"},
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    output = dataset_converter(example)

    # Should return a single dict (not a list)
    assert isinstance(output, dict)
    assert "_mask_history_sample" not in output
    # For sharegpt format, prompt includes all messages except the last one
    # So [user1, assistant1, user2, assistant2] -> prompt=[user1, assistant1, user2], response=[assistant2]
    assert len(output["_prompt"]) == 3  # user1, assistant1, user2
    assert len(output["_response"]) == 1  # assistant2


def test_mask_history_sample_validation():
    """Test that mask_history_sample and max_human_steps must be set together."""
    # Test that both default values work
    dataset_attr1 = DatasetAttr("file", "test_dataset")
    dataset_attr1.mask_history_sample = False
    dataset_attr1.max_human_steps = -1
    # Should not raise error

    # Test that both set values work
    dataset_attr2 = DatasetAttr("file", "test_dataset")
    dataset_attr2.mask_history_sample = True
    dataset_attr2.max_human_steps = 2
    # Should not raise error

    # Test that mismatched values raise error
    # This should have been caught during join(), but let's test the join method
    attr = DatasetAttr("file", "test_dataset")
    with pytest.raises(ValueError, match="mask_history_sample and max_human_steps must be set together"):
        attr.join({"mask_history_sample": True, "max_human_steps": -1})

    with pytest.raises(ValueError, match="mask_history_sample and max_human_steps must be set together"):
        attr.join({"mask_history_sample": False, "max_human_steps": 2})

    # Test that max_human_steps must be >= 1 when set
    with pytest.raises(ValueError, match="max_human_steps must be >= 1"):
        attr.join({"mask_history_sample": True, "max_human_steps": 0})


def test_mask_history_sample_single_turn():
    """Test that single turn conversation works correctly."""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.mask_history_sample = True
    dataset_attr.max_human_steps = 2

    data_args = DataArguments()

    # Single turn conversation
    example = {
        "conversations": [
            {"from": "human", "value": "user1"},
            {"from": "gpt", "value": "assistant1"},
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    outputs = dataset_converter(example)

    # Should return a list with 1 sample
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert outputs[0]["_prompt"] == [{"role": Role.USER.value, "content": "user1"}]
    assert outputs[0]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant1"}]
    assert outputs[0]["_mask_history_sample"] is True


def test_mask_history_sample_max_human_steps_one():
    """Test that max_human_steps=1 works correctly."""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.mask_history_sample = True
    dataset_attr.max_human_steps = 1

    data_args = DataArguments()

    # Create example with 3 turns
    example = {
        "conversations": [
            {"from": "human", "value": "user1"},
            {"from": "gpt", "value": "assistant1"},
            {"from": "human", "value": "user2"},
            {"from": "gpt", "value": "assistant2"},
            {"from": "human", "value": "user3"},
            {"from": "gpt", "value": "assistant3"},
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    outputs = dataset_converter(example)

    # Should return 3 samples
    assert isinstance(outputs, list)
    assert len(outputs) == 3

    # Each sample should have at most 1 user message in prompt
    for i, output in enumerate(outputs):
        prompt = output["_prompt"]
        user_count = sum(1 for msg in prompt if msg["role"] == Role.USER.value)
        assert user_count <= 1, f"Sample {i} has {user_count} user messages, should be <= 1"
        assert output["_mask_history_sample"] is True

    # First sample: [user1] -> [assistant1]
    assert outputs[0]["_prompt"] == [{"role": Role.USER.value, "content": "user1"}]
    assert outputs[0]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant1"}]

    # Second sample: [assistant1, user2] -> [assistant2]
    assert len(outputs[1]["_prompt"]) == 2
    assert outputs[1]["_prompt"][0]["role"] == Role.ASSISTANT.value
    assert outputs[1]["_prompt"][1]["role"] == Role.USER.value
    assert outputs[1]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant2"}]

    # Third sample: [assistant1, assistant2, user3] -> [assistant3]
    # When max_human_steps=1, we keep only user3, but we also keep all assistant messages before it
    assert len(outputs[2]["_prompt"]) == 3
    assert outputs[2]["_prompt"][0]["role"] == Role.ASSISTANT.value
    assert outputs[2]["_prompt"][0]["content"] == "assistant1"
    assert outputs[2]["_prompt"][1]["role"] == Role.ASSISTANT.value
    assert outputs[2]["_prompt"][1]["content"] == "assistant2"
    assert outputs[2]["_prompt"][2]["role"] == Role.USER.value
    assert outputs[2]["_prompt"][2]["content"] == "user3"
    assert outputs[2]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant3"}]


def test_mask_history_sample_max_human_steps_large():
    """Test that large max_human_steps works correctly."""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.mask_history_sample = True
    dataset_attr.max_human_steps = 10  # Large value

    data_args = DataArguments()

    # Create example with 3 turns
    example = {
        "conversations": [
            {"from": "human", "value": "user1"},
            {"from": "gpt", "value": "assistant1"},
            {"from": "human", "value": "user2"},
            {"from": "gpt", "value": "assistant2"},
            {"from": "human", "value": "user3"},
            {"from": "gpt", "value": "assistant3"},
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    outputs = dataset_converter(example)

    # Should return 3 samples
    assert isinstance(outputs, list)
    assert len(outputs) == 3

    # First sample: [user1] -> [assistant1]
    assert outputs[0]["_prompt"] == [{"role": Role.USER.value, "content": "user1"}]
    assert outputs[0]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant1"}]

    # Second sample: [user1, assistant1, user2] -> [assistant2]
    assert len(outputs[1]["_prompt"]) == 3
    assert outputs[1]["_prompt"][0]["role"] == Role.USER.value
    assert outputs[1]["_prompt"][1]["role"] == Role.ASSISTANT.value
    assert outputs[1]["_prompt"][2]["role"] == Role.USER.value
    assert outputs[1]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant2"}]

    # Third sample: [user1, assistant1, user2, assistant2, user3] -> [assistant3]
    assert len(outputs[2]["_prompt"]) == 5
    assert outputs[2]["_prompt"][0]["role"] == Role.USER.value
    assert outputs[2]["_prompt"][1]["role"] == Role.ASSISTANT.value
    assert outputs[2]["_prompt"][2]["role"] == Role.USER.value
    assert outputs[2]["_prompt"][3]["role"] == Role.ASSISTANT.value
    assert outputs[2]["_prompt"][4]["role"] == Role.USER.value
    assert outputs[2]["_response"] == [{"role": Role.ASSISTANT.value, "content": "assistant3"}]

