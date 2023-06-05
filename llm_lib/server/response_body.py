from typing import List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal

from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict

class CompletionChoice(TypedDict):
    text: str
    index: int
    finish_reason: Optional[str]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: List[int]
    total_tokens: List[int]

class CompetionResponseType(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    # usage: Optional[CompletionUsage]

CompletionRepsoneBody = create_model_from_typeddict(CompetionResponseType)