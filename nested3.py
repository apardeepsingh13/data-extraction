from typing import List, Optional
from pydantic import BaseModel, Field


# Define OutputFormat class
class OutputFormat(BaseModel):
    sources: str = Field(
        ..., description="The raw transcript / span you could cite to justify the choice."
    )
    content: str = Field(..., description="The chosen value.")


# Define BackgroundInfo class
class BackgroundInfo(BaseModel):
    factoid: OutputFormat = Field(
        ..., description="Important factoid about the member."
    )
    professions: Optional[List[str]] = Field(default_factory=list)
    why: str = Field(..., description="Why this is important.")


# Define Moment class
class Moment(BaseModel):
    quote: str = Field(..., description="The relevant quote from the transcript.")
    description: str = Field(..., description="A description of the moment.")
    expressed_preference: OutputFormat = Field(
        ..., description="The preference expressed in the moment."
    )


# Define KeyMoments class
class KeyMoments(BaseModel):
    topic: str = Field(..., description="The topic of the key moments.")
    happy_moments: List[Moment] = Field(
        default_factory=list, description="A list of key moments related to the topic."
    )
    tense_moments: List[Moment] = Field(
        default_factory=list, description="Moments where things were a bit tense."
    )
    sad_moments: List[Moment] = Field(
        default_factory=list, description="Moments where everyone was downtrodden."
    )
    background_info: List[BackgroundInfo] = Field(
        default_factory=list, description="Background information for the key moments."
    )
    moments_summary: str = Field(..., description="A summary of the key moments.")


# Define Member class
class Member(BaseModel):
    name: OutputFormat = Field(..., description="The name of the member.")
    role: str = Field(..., description="The role of the member.")
    age: Optional[int] = Field(None, description="The age of the member.")
    background_details: List[BackgroundInfo] = Field(
        default_factory=list, description="A list of background details about the member."
    )


# Define InsightfulQuote class
class InsightfulQuote(BaseModel):
    quote: OutputFormat = Field(
        ..., description="An insightful quote from the transcript."
    )
    speaker: str = Field(..., description="The name of the speaker who said the quote.")
    analysis: str = Field(
        ..., description="An analysis of the quote and its significance."
    )


# Define TranscriptMetadata class
class TranscriptMetadata(BaseModel):
    title: str = Field(..., description="The title of the transcript.")
    location: OutputFormat = Field(
        ..., description="The location where the interview took place."
    )
    duration: str = Field(..., description="The duration of the interview.")


# Define TranscriptSummary class
class TranscriptSummary(BaseModel):
    metadata: TranscriptMetadata = Field(
        ..., description="Metadata about the transcript."
    )
    participants: List[Member] = Field(
        default_factory=list, description="A list of participants in the interview."
    )
    key_moments: List[KeyMoments] = Field(
        default_factory=list, description="A list of key moments from the interview."
    )
    insightful_quotes: List[InsightfulQuote] = Field(
        default_factory=list, description="A list of insightful quotes from the interview."
    )
    overall_summary: str = Field(
        ..., description="An overall summary of the interview."
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="A list of next steps or action items based on the interview.",
    )
    other_stuff: List[OutputFormat] = Field(
        default_factory=list, description="Additional information."
    )


# Create default data for each model
metadata = TranscriptMetadata(
    title="Sample Transcript Title",
    location=OutputFormat(sources="Source 1", content="Location 1"),
    duration="1 hour"
)

participants = [
    Member(
        name=OutputFormat(sources="Source 2", content="John Doe"),
        role="Interviewee",
        age=30,
        background_details=[
            BackgroundInfo(
                factoid=OutputFormat(sources="Source 3", content="Key Fact"),
                professions=["Engineer"],
                why="Relevant to the interview topic"
            )
        ]
    )
]

key_moments = [
    KeyMoments(
        topic="Project Milestones",
        happy_moments=[
            Moment(
                quote="This was a great success!",
                description="Discussing the project launch.",
                expressed_preference=OutputFormat(sources="Source 4", content="Positive Feedback")
            )
        ],
        moments_summary="Key highlights and milestones."
    )
]

insightful_quotes = [
    InsightfulQuote(
        quote=OutputFormat(sources="Source 5", content="Key quote about teamwork."),
        speaker="Jane Smith",
        analysis="Highlights the importance of collaboration."
    )
]

# Create the final TranscriptSummary
transcript_summary = TranscriptSummary(
    metadata=metadata,
    participants=participants,
    key_moments=key_moments,
    insightful_quotes=insightful_quotes,
    overall_summary="This is a summary of the interview.",
    next_steps=["Follow up with participants", "Plan next meeting"],
    other_stuff=[
        OutputFormat(sources="Source 6", content="Additional context.")
    ]
)

# Serialize and print JSON
print(transcript_summary.json(indent=2))
