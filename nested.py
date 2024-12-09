from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from transcript import transcript



class OutputFormat(BaseModel):
    sources: str = Field(
        ...,
        description="The raw transcript / span you could cite to justify the choice.",
    )
    content: str = Field(..., description="The chosen value.")


class Moment(BaseModel):
    quote: str = Field(..., description="The relevant quote from the transcript.")
    description: str = Field(..., description="A description of the moment.")
    expressed_preference: OutputFormat = Field(
        ..., description="The preference expressed in the moment."
    )


class BackgroundInfo(BaseModel):
    factoid: Optional[OutputFormat] = Field(
        ..., description="Important factoid about the member."
    )
    professions: Optional[list]
    why: str = Field(..., description="Why this is important.")


class KeyMoments(BaseModel):
    topic: Optional[str] = Field(..., description="The topic of the key moments.")
    happy_moments: Optional[List[Moment]] = Field(
        ..., description="A list of key moments related to the topic."
    )
    tense_moments: Optional[List[Moment]] = Field(
        ..., description="Moments where things were a bit tense."
    )
    sad_moments: Optional[List[Moment]] = Field(
        ..., description="Moments where things where everyone was downtrodden."
    )
    background_info: Optional[list[BackgroundInfo]]
    moments_summary: str = Field(..., description="A summary of the key moments.")


class Member(BaseModel):
    name: OutputFormat = Field(..., description="The name of the member.")
    role: Optional[str] = Field(None, description="The role of the member.")
    age: Optional[int] = Field(None, description="The age of the member.")
    background_details: Optional[List[BackgroundInfo]] = Field(
        ..., description="A list of background details about the member."
    )


class InsightfulQuote(BaseModel):
    quote: OutputFormat = Field(
        ..., description="An insightful quote from the transcript."
    )
    speaker: str = Field(..., description="The name of the speaker who said the quote.")
    analysis: str = Field(
        ..., description="An analysis of the quote and its significance."
    )


class TranscriptMetadata(BaseModel):
    title: str = Field(..., description="The title of the transcript.")
    location: OutputFormat = Field(
        ..., description="The location where the interview took place."
    )
    duration: str = Field(..., description="The duration of the interview.")


class TranscriptSummary(BaseModel):
    metadata: TranscriptMetadata = Field(
        ..., description="Metadata about the transcript."
    )
    participants: Optional[List[Member]] = Field(
        ..., description="A list of participants in the interview."
    )
    key_moments: Optional[List[KeyMoments]] = Field(
        ..., description="A list of key moments from the interview."
    )
    insightful_quotes: List[InsightfulQuote] = Field(
        ..., description="A list of insightful quotes from the interview."
    )
    overall_summary: str = Field(
        ..., description="An overall summary of the interview."
    )
    next_steps: List[str] = Field(
        ..., description="A list of next steps or action items based on the interview."
    )
    other_stuff: List[OutputFormat]
