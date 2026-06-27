"""Enums for TensorEval — environment types, modalities, grader types, difficulty.

Based on research across Verifiers, HUD, Inspect AI, ART, and Tinker.
"""

from enum import Enum


class EnvType(str, Enum):
    """Environment interaction types.

    Defines HOW the agent interacts with the environment.
    """

    SINGLE_TURN = "single_turn"
    """Q&A, single response evaluation. Agent responds once."""

    MULTI_TURN = "multi_turn"
    """Conversational, iterative interactions. Agent responds multiple times."""

    TOOL_USE = "tool_use"
    """Function-calling with idempotent tools. Agent calls Python functions."""

    STATEFUL_TOOL = "stateful_tool"
    """Tools with per-session state (sandbox ID, DB connections)."""

    SANDBOX = "sandbox"
    """Isolated Docker container execution. Agent runs code in container."""

    PYTHON_REPL = "python_repl"
    """Persistent Python REPL. Agent executes code that persists state."""

    GYM = "gym"
    """Gymnasium-compatible step/reset protocol. For game/simulation evals."""

    MCP = "mcp"
    """Model Context Protocol server integration. Agent uses MCP tools."""

    BROWSER = "browser"
    """Browser automation (CDP protocol). Agent controls a browser."""

    COMPUTER = "computer"
    """Desktop computer control (VNC/RFB). Agent controls mouse/keyboard."""

    ROBOT = "robot"
    """Robot control (action/observation loop). Agent controls a robot."""

    VOICE = "voice"
    """Voice/audio interaction. Agent processes audio input/output."""

    CLI_AGENT = "cli_agent"
    """Agent code in sandboxes with API interception."""

    CUSTOM = "custom"
    """Custom environment with user-defined interaction pattern."""


class Modality(str, Enum):
    """Input/output modalities.

    Defines WHAT data the agent processes.
    """

    TEXT = "text"
    """Text-only input and output."""

    IMAGE = "image"
    """Image input (screenshots, photos, diagrams)."""

    AUDIO = "audio"
    """Audio input/output (speech, sounds)."""

    VIDEO = "video"
    """Video input (camera feeds, screen recordings)."""

    DOCUMENT = "document"
    """Document input (PDF, Word, Excel)."""

    MULTIMODAL = "multimodal"
    """Multiple modalities combined (text + image + audio)."""

    CODE = "code"
    """Code input/output (Python, JavaScript, etc.)."""

    STRUCTURED = "structured"
    """Structured data (JSON, XML, CSV, database queries)."""


class GraderType(str, Enum):
    """Grader/scorer types.

    Defines HOW responses are scored.
    """

    RUBRIC = "rubric"
    """Rule-based scoring with weighted rubrics. Deterministic."""

    AGENT = "agent"
    """LLM-as-judge scoring. Uses another model to judge responses."""

    RULER = "ruler"
    """Zero-config relative ranking via LLM. No rubrics needed."""

    EXACT_MATCH = "exact_match"
    """Exact string match against reference answer."""

    FUZZY_MATCH = "fuzzy_match"
    """Fuzzy string match (allows minor variations)."""

    MATH = "math"
    """Math answer extraction and comparison (boxed format)."""

    CODE_EXEC = "code_exec"
    """Execute code and check output. Tests code correctness."""

    CUSTOM = "custom"
    """User-defined scoring function."""


class Difficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    """Well-documented, straightforward. Strict scoring."""

    MEDIUM = "medium"
    """Requires combining multiple sources/steps. Standard scoring."""

    HARD = "hard"
    """Complex multi-step, conflicting info. Generous partial credit."""


class Provider(str, Enum):
    """Model API providers."""

    OPENAI = "openai"
    """OpenAI API (GPT-4, GPT-4o, etc.)"""

    ANTHROPIC = "anthropic"
    """Anthropic API (Claude)"""

    MIMO = "mimo"
    """Xiaomi Mimo API"""

    TOGETHER = "together"
    """Together AI API"""

    LOCAL = "local"
    """Local model (vLLM, Ollama, etc.)"""

    CUSTOM = "custom"
    """Custom API endpoint"""
