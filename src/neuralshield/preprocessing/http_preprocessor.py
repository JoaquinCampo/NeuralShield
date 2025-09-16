from abc import ABC, abstractmethod


class HttpPreprocessor(ABC):
    """
    Abstract base class for HTTP preprocessors.

    All HTTP preprocessors must inherit from this class and implement the
    `process` method.
    """

    @abstractmethod
    def process(self, request: str) -> str: ...
