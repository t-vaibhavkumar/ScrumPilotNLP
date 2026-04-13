"""
ScrumExtractorAgent — Uses a LangChain chain (Gemini LLM + PydanticOutputParser)
to parse a diarized meeting transcript into a list of scrum actions.

Now enhanced with an NLP preprocessing layer.
"""

import os
import json
import re
from typing import List, Optional, Literal, Dict

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


# ── NLP PREPROCESSING ─────────────────────────────────────────────────────────

def parse_transcript(transcript: str) -> List[Dict]:
    """Parse diarized transcript into speaker-text pairs."""
    lines = transcript.split("\n")
    parsed = []

    pattern = r"^(.*?):\s*(.*)$"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            parsed.append({"speaker": speaker, "text": text})

    return parsed


def split_sentences(parsed_data: List[Dict]) -> List[Dict]:
    """Split speaker text into individual sentences."""
    results = []

    for entry in parsed_data:
        speaker = entry["speaker"]
        text = entry["text"]

        sentences = re.split(r"[.!?]+", text)

        for s in sentences:
            s = s.strip()
            if s:
                results.append({
                    "speaker": speaker,
                    "sentence": s
                })

    return results


# ── Structured output schemas ─────────────────────────────────────────────────

class ScrumAction(BaseModel):
    action: Literal[
        "create_task",
        "complete_task",
        "update_status",
        "assign_task",
        "add_comment",
    ]
    summary: str
    description: Optional[str] = None
    assignee: Optional[str] = None
    status: Optional[str] = None
    comment: Optional[str] = None


class ScrumActionList(BaseModel):
    actions: List[ScrumAction]


# ── Agent class ───────────────────────────────────────────────────────────────

class ScrumExtractorAgent:
    """LangChain-powered agent with NLP preprocessing."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0,
        )

        parser = PydanticOutputParser(pydantic_object=ScrumActionList)
        format_instructions = parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a Scrum Action Extractor.\n\n"
                    "You will receive PRE-PROCESSED structured sentences.\n"
                    "Each item has speaker + sentence.\n\n"
                    "Your job:\n"
                    "- Convert them into structured scrum actions\n"
                    "- Use speaker as assignee when relevant\n\n"
                    "Rules:\n"
                    "- 'complete_task' → task finished\n"
                    "- 'update_status' → in progress or status change\n"
                    "- 'assign_task' → assigning work\n"
                    "- 'create_task' → new task mentioned\n"
                    "- 'add_comment' → explicit comment\n\n"
                    "Output format:\n{format_instructions}"
                ),
            ),
            ("human", "Structured sentences:\n\n{transcript}"),
        ])

        self.chain = prompt | llm | parser
        self._format_instructions = format_instructions

    def extract_actions(self, transcript: str) -> List[dict]:
        """
        NLP → structured sentences → LLM → actions
        """

        # ── Step 1: NLP preprocessing ───────────────────────────
        parsed = parse_transcript(transcript)
        sentences = split_sentences(parsed)

        print("\n[NLP DEBUG] Parsed Sentences:")
        for s in sentences:
            print(s)

        # Convert structured sentences → JSON string
        structured_input = json.dumps(sentences, indent=2)

        # ── Step 2: LLM extraction (minimal role now) ───────────
        result: ScrumActionList = self.chain.invoke({
            "transcript": structured_input,
            "format_instructions": self._format_instructions,
        })

        return [action.model_dump(exclude_none=True) for action in result.actions]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    here = os.path.dirname(__file__)
    transcript_path = os.path.join(here, "example_transcript.txt")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    agent = ScrumExtractorAgent()
    actions = agent.extract_actions(transcript)

    print("\nFinal Actions:\n")
    print(json.dumps(actions, indent=2))