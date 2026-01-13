from abc import ABC
from typing import Any
from simple_xmm.utils.registry import register_template


class BaseFormatter(ABC):
    def is_valid(self, raw_sample: dict[str, Any]) -> bool:
        return True

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict]:
        """Format the sample for supervised training.

        Args:
            raw_sample (dict[str, Any]): The raw sample from the dataset.

        Example:
            >>> self.format_supervised_sample({'instruction': 'Write a story', 'output': 'Once upon a time, there was a cat.'})
            ([{'role': 'user', 'content': 'Write a story'}, {'role': 'assistant', 'content': 'Once upon a time, there was a cat.'}], '')

        Returns:
            tuple[list[dict[str, Any]], dict]: The formatted sample.
        """
        return [], {}


@register_template("Alpaca")
class Alpaca(BaseFormatter):
    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = " ".join((raw_sample["instruction"], raw_sample["input"]))
        response = raw_sample["output"]
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ], {}
