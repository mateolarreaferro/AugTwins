"""
Interactive GPT-powered code assistant.

Run from the repo root to explore your codebase via GPT.
Type “chao” to exit.
"""
import argparse
from pathlib import Path
from typing import Dict

from core.llm_utils import gen_oai  

try:
    import readline         
except ImportError:             
    import pyreadline3 as readline



# This is what determines what context the agent has
SOURCE_FILES = [
    "app.py",
    "settings.py",
    "core/llm_utils.py",
    "core/agent.py",
    "core/memory_utils.py",
    "agents/Lars/lars.py",
]


def load_code(files) -> Dict[str, str]:
    """Read each file and return a dict {filename: contents}."""
    code_map: Dict[str, str] = {}
    for fn in files:
        path = Path(fn)
        if path.exists():
            code_map[fn] = path.read_text(encoding="utf-8")
        else:
            code_map[fn] = f"# ERROR: '{fn}' not found\n"
    return code_map


def make_system_prompt(code_map: Dict[str, str]) -> str:
    """Concatenate all file contents under headers for the system prompt."""
    sections = [
        f"### FILE: {name}\n{content}\n" for name, content in code_map.items()
    ]
    return (
        "You are a helpful assistant with full knowledge of the codebase. "
        "Answer questions by referring only to the provided code.\n\n"
        + "\n".join(sections)
    )


def ask_codebase(
    history: list[dict], code_map: Dict[str, str], question: str,
    model: str, temp: float
) -> str:
    """Append user question, call GPT, append assistant reply, return reply."""
    if not history: 
        history.append({"role": "system", "content": make_system_prompt(code_map)})

    history.append({"role": "user", "content": question})
    reply = gen_oai(history, model=model, temperature=temp)
    history.append({"role": "assistant", "content": reply})
    return reply


def repl(code_map: Dict[str, str], model: str, temp: float) -> None:
    """Launch interactive REPL.  Type 'chao' to quit."""
    print("💻  Code Assistant — ask me about your code (type 'chao' to exit)\n")
    history: list[dict] = []
    try:
        while True:
            user_q = input("🖥️  Question: ").strip()
            if not user_q:
                continue
            if user_q.lower() == "chao":
                print("\n👋  Chao!  Goodbye.\n")
                break

            print("\n🤖  Thinking…\n")
            answer = ask_codebase(history, code_map, user_q, model, temp)
            print(answer)
            print("\n" + "-" * 60 + "\n")
    except (EOFError, KeyboardInterrupt):
        print("\n👋  Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-powered code assistant")
    parser.add_argument(
        "-m", "--model", default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "-t", "--temp", type=float, default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    args = parser.parse_args()

    code_map = load_code(SOURCE_FILES)
    repl(code_map, args.model, args.temp)


if __name__ == "__main__":
    main()
