# Multi-Agent AI Meeting Assistant

## Overview

The Multi-Agent AI Meeting Assistant converts a speaker-labelled meeting transcript
into a reviewed report containing:

- Main topics and a factual summary
- Explicit decisions with transcript evidence
- Action items with owners, deadlines, evidence, and provenance
- Human clarification for missing owners or deadlines
- Review findings, bounded corrections, and final review status

The application is a console-based Python 3.11+ capstone. Version 1.1.2 supports
Ollama, Gemini, and OpenAI through one provider-neutral LangChain interface.

## Architecture

All implementation remains in `meeting_assistant.py`. Provider selection is isolated
behind one factory:

```text
Configuration / runtime selection
              |
              v
         create_llm()
              |
     +--------+--------+
     |        |        |
  Ollama   Gemini   OpenAI
     |        |        |
     +--------+--------+
              |
              v
       BaseChatModel
              |
              v
       Same LangGraph workflow
```

Agent functions and graph nodes receive only `BaseChatModel`. There are no
provider-specific agents, schemas, parsers, or workflows.

Transcript text is treated as untrusted data. System instructions and transcript
content remain separated in `SystemMessage` and `HumanMessage` objects.

## Features

- UTF-8 transcript-file input and a built-in demo fixture
- Strict `Speaker: message` parsing with physical-line error reporting
- Pydantic contracts for every structured agent response
- Evidence-line validation against the normalized transcript
- Provider-native structured output through `with_structured_output()`
- Human clarification with transcript/human/unresolved provenance
- Independent review, one bounded correction pass, and final re-review
- Sanitized provider errors and DEBUG-only diagnostic detail
- Visible pre-workflow fallback from Gemini or OpenAI to Ollama
- Deterministic final-report rendering without an additional LLM call

## Technology Stack

- Python 3.11+
- Pydantic and pydantic-settings
- LangChain Core
- LangGraph
- langchain-ollama
- langchain-google-genai
- langchain-openai
- python-dotenv

Dependency ranges are listed in `requirements.txt`. The file is intentionally not a
reproducible lockfile.

## Workflow

```text
Transcript loading and parsing
            |
            v
      Summary Agent
            |
            v
    Action Item Agent
            |
            v
       Review Agent
            |
            v
 Human clarification when required
            |
            v
 One correction pass when required
            |
            v
        Re-review
            |
            v
   Final Meeting Report
```

Routing is deterministic. Missing action-item fields can trigger clarification.
Supported semantic review findings can trigger at most one correction pass. A model
is selected and validated before the graph starts; the application never switches
providers during a running workflow.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python meeting_assistant.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The default command is an offline preview. It does not create or invoke an LLM.

## LLM Provider Configuration

Copy `.env.example` to `.env` for local configuration, or set environment variables
directly. Process environment variables override `.env`, which overrides defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `gemini`, or `openai` |
| `OLLAMA_MODEL` | `llama3.2:latest` | Local Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model; blank uses the default |
| `GOOGLE_API_KEY` | unset | Required when Gemini is active |
| `OPENAI_MODEL` | `gpt-5-mini` | OpenAI model; blank uses the default |
| `OPENAI_API_KEY` | unset | Required when OpenAI is active |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, or CRITICAL |

Credentials use `SecretStr`, are excluded from configuration serialization, and are
never printed by application code.

### Provider selection

Use `--provider` for deterministic execution:

```sh
python meeting_assistant.py --run-final --sample --provider ollama
python meeting_assistant.py --run-final --sample --provider gemini
python meeting_assistant.py --run-final --sample --provider openai
```

On an interactive terminal without an explicit provider setting, the application
offers a bounded provider menu. `--select-provider` forces that menu. Precedence is:

1. `--provider`
2. `--select-provider`
3. Explicit `LLM_PROVIDER`
4. Interactive terminal selection
5. Configured default

Non-interactive execution never waits for a provider-selection prompt.

### Provider readiness and fallback

Before a workflow starts, the selected provider must pass basic connectivity and a
minimal structured-output probe. If Gemini or OpenAI lacks a credential or fails
readiness, the application displays a sanitized reason and attempts Ollama once.
The execution header distinguishes the requested provider from the active provider.

If Ollama fallback also fails, the workflow exits cleanly. Failures after the graph
starts do not trigger fallback or restart completed agents.

## Running the Application

Run the final reviewed report with a transcript file:

```sh
python meeting_assistant.py \
  --run-final \
  --transcript sample_meeting.txt \
  --provider ollama
```

Run with the built-in demo transcript:

```sh
python meeting_assistant.py --run-final --sample --provider ollama
```

Supported modes:

| Option | Behavior |
| --- | --- |
| `--check-llm` | Test connectivity and structured output independently |
| `--run-summary` | Run only summary and decision extraction |
| `--run-actions` | Run only action-item extraction |
| `--run-review` | Generate candidates and run independent review |
| `--run-clarification` | Extract actions and request missing fields |
| `--run-workflow` | Display detailed graph/debug output |
| `--run-final` | Render the polished final meeting report |

Diagnostic example:

```sh
LOG_LEVEL=DEBUG python meeting_assistant.py --check-llm --provider ollama
```

Connectivity and structured output are tested independently and reported as
`READY` or `FAILED`. DEBUG logs include the failed stage, exception class, and a
sanitized message.

## Transcript Format

Transcript files must be UTF-8 text with one speaker-labelled entry per physical
line:

```text
Ravi: I will complete the API testing by Friday.
Priya: I will update the design document by Monday.
Ravi: Someone needs to review security.
```

Speaker labels and messages must be nonempty. Blank lines are ignored. Text after
the first colon remains part of the message, so URLs and times are preserved.

`--transcript` and `--sample` are mutually exclusive. When neither is supplied, the
built-in sample remains the backward-compatible default.

## Example

The repository-level `sample_meeting.txt` contains the same demonstration meeting
as the built-in fixture. The sample deliberately includes two incomplete action
items so the human-clarification path remains visible. Clarification responses are
entered at runtime and are never hardcoded.

## Error Handling and Limitations

Expected configuration, transcript, provider, agent, and workflow errors are shown
without Python tracebacks. Provider errors do not include keys, tokens, authorization
headers, or request credentials.

LLM output remains model-dependent. A provider can pass readiness and later fail
because of capacity, quota, networking, or model behavior. Such failures stop the
current workflow without switching providers midway. Review quality can also vary by
model; the application does not hardcode corrections to improve a demo result.

## Google Colab

Upload `meeting_assistant.py`, `requirements.txt`, and the transcript file. Install
dependencies and select Gemini or OpenAI deterministically:

```python
%pip install -r requirements.txt
!python meeting_assistant.py --run-final --transcript meeting.txt --provider gemini
```

Store `GOOGLE_API_KEY` or `OPENAI_API_KEY` in Colab Secrets and transfer it to the
matching environment variable without printing it. No Colab-specific transcript or
workflow implementation is required.

## Project Structure

```text
meeting_assistant.py  Single-file application
sample_meeting.txt    Demo transcript fixture
.env.example          Safe configuration template
requirements.txt      Bounded dependency ranges
README.md             Setup and architecture documentation
```

Local `.env` files, virtual environments, caches, IDE metadata, logs, and build
artifacts are excluded through `.gitignore`.
