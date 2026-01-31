#!/usr/bin/env python3
"""
UBIK Somatic Node - Golden Set Test Runner

Runs all Golden Set prompts through the RAG pipeline with interactive
human evaluation. Part of the Phase 3 evaluation framework (Section 7.3).

Modes:
    (default)        Interactive mode - score each response as generated
    --batch, -b      Generate all responses, score later

Interactive mode enables:
    - Real-time scoring (1-5) with rationale
    - Ideal response capture for DPO training
    - Optional memory storage for immediate RAG improvement

Bilingual Support:
    Prompts are randomly presented in English or Spanish to test voice
    authenticity across languages.

Usage:
    # Interactive mode (default)
    python golden_set_runner.py

    # Batch mode (score later)
    python golden_set_runner.py --batch

    # Force specific language:
    python golden_set_runner.py --lang en

Author: UBIK Project
Version: 3.0.0 (Phase 3 - Interactive + DPO)
"""

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGService

# =============================================================================
# Language Selection
# =============================================================================

Language = Literal["en", "es"]

# Simple translations for common prompts (fallback when prompt_es not provided)
PROMPT_TRANSLATIONS: Dict[str, str] = {
    "What does family mean to you?": "¿Qué significa la familia para ti?",
    "What do you want your grandchildren to remember about you?": "¿Qué quieres que tus nietos recuerden de ti?",
    "What do you believe about authenticity?": "¿Qué crees sobre la autenticidad?",
    "What matters most to you in life?": "¿Qué es lo que más te importa en la vida?",
    "What has life taught you about love?": "¿Qué te ha enseñado la vida sobre el amor?",
    "How have your beliefs changed over time?": "¿Cómo han cambiado tus creencias con el tiempo?",
    "What advice would you give about facing difficult times?": "¿Qué consejo darías para enfrentar tiempos difíciles?",
    "How should I think about making important decisions?": "¿Cómo debería pensar sobre tomar decisiones importantes?",
    "Who are you, at your core?": "¿Quién eres, en tu esencia?",
    "What makes you, you?": "¿Qué te hace ser tú?",
}

SCORING_RUBRIC = """
┌─────────────────────────────────────────────────────────────┐
│                    SCORING RUBRIC                           │
├─────────────────────────────────────────────────────────────┤
│  5 - Excellent: Sounds exactly like Gines would say it      │
│  4 - Good: Captures voice well with minor imperfections     │
│  3 - Acceptable: Generic but not wrong, missing warmth      │
│  2 - Poor: Doesn't sound like Gines, wrong tone             │
│  1 - Failure: Completely wrong, shows reasoning, or error   │
└─────────────────────────────────────────────────────────────┘
"""


def select_language(force_lang: Optional[Language] = None) -> Language:
    """Randomly select English or Spanish, or use forced language."""
    if force_lang:
        return force_lang
    return random.choice(["en", "es"])


def get_prompt_in_language(item: Dict[str, Any], lang: Language) -> str:
    """Get the prompt text in the specified language."""
    if lang == "es" and "prompt_es" in item:
        return item["prompt_es"]
    if lang == "en" and "prompt_en" in item:
        return item["prompt_en"]

    base_prompt = item.get("prompt", "")
    if lang == "en":
        return base_prompt

    if base_prompt in PROMPT_TRANSLATIONS:
        return PROMPT_TRANSLATIONS[base_prompt]

    return f"[NO ES TRANSLATION] {base_prompt}"


# =============================================================================
# Interactive Scoring
# =============================================================================


def get_score_input() -> Optional[int]:
    """Get score input from user (1-5 or 's' to skip)."""
    while True:
        try:
            value = input("Score (1-5, or 's' to skip): ").strip().lower()
            if value == 's' or value == '':
                return None
            score = int(value)
            if 1 <= score <= 5:
                return score
            print("  Please enter 1-5 or 's' to skip")
        except ValueError:
            print("  Please enter 1-5 or 's' to skip")


def get_multiline_input(prompt: str) -> str:
    """Get multiline input, ending with empty line or Ctrl+D."""
    print(prompt)
    print("  (Enter your response, then press Enter twice to finish)")
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 1:  # Single empty line to finish
                    break
                lines.append("")
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


async def store_as_memory(
    content: str,
    category: str,
    prompt: str,
) -> Optional[str]:
    """Store ideal response as a semantic memory."""
    try:
        from mcp_client import HippocampalClient

        async with HippocampalClient() as client:
            result = await client.store_semantic(
                content=content,
                knowledge_type="value",
                category=category,
                confidence=0.9,
                stability="stable",
                source="golden_set_feedback",
            )
            if result.get("status") == "success":
                memory_id = result.get("id", "unknown")
                return memory_id
            else:
                print(f"  ⚠ Failed to store: {result.get('message', 'Unknown error')}")
                return None
    except Exception as e:
        print(f"  ⚠ Memory storage error: {e}")
        return None


def save_dpo_pair(
    dpo_file: Path,
    prompt: str,
    rejected: str,
    chosen: str,
    category: str,
    score: int,
) -> None:
    """Append a DPO training pair to the DPO file."""
    pair = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "original_score": score,
        "timestamp": datetime.now().isoformat(),
    }

    # Load existing pairs or create new list
    if dpo_file.exists():
        with open(dpo_file) as f:
            data = json.load(f)
    else:
        data = {"metadata": {"created": datetime.now().isoformat()}, "pairs": []}

    data["pairs"].append(pair)

    with open(dpo_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def interactive_score(
    result: Dict[str, Any],
    dpo_file: Path,
    prompt_num: int,
    total_prompts: int,
) -> Dict[str, Any]:
    """
    Interactively score a response and optionally collect ideal response.

    Returns the result dict updated with score, rationale, ideal_response.
    """
    prompt = result["prompt"]
    response = result["response"]
    category = result["category"]

    print(f"\n{'═' * 70}")
    print(f" [{prompt_num}/{total_prompts}] {category.upper()}")
    print(f"{'═' * 70}")
    print(f"\n📝 PROMPT:\n{prompt}\n")
    print(f"{'─' * 70}")
    print(f"\n🤖 MODEL RESPONSE:\n{response}\n")
    print(f"{'─' * 70}")

    # Get score
    print(SCORING_RUBRIC)
    score = get_score_input()

    if score is None:
        print("  Skipped")
        return result

    result["score"] = score

    # Get rationale
    rationale = input("Rationale (brief, or Enter to skip): ").strip()
    if rationale:
        result["score_rationale"] = rationale

    # For low scores (1-3), offer to provide ideal response
    if score <= 3:
        print(f"\n{'─' * 70}")
        print("💡 Low score detected. Would you like to provide an ideal response?")
        provide_ideal = input("   Provide ideal response? [y/N]: ").strip().lower()

        if provide_ideal == 'y':
            ideal = get_multiline_input("\n✍ YOUR IDEAL RESPONSE:")

            if ideal:
                result["ideal_response"] = ideal

                # Option to save as memory
                save_mem = input("\n   Save as memory for future RAG? [y/N]: ").strip().lower()
                if save_mem == 'y':
                    memory_id = await store_as_memory(ideal, category, prompt)
                    if memory_id:
                        print(f"   ✓ Saved as memory: {memory_id}")
                        result["memory_id"] = memory_id

                # Option to save for DPO
                save_dpo = input("   Save for DPO training? [Y/n]: ").strip().lower()
                if save_dpo != 'n':
                    save_dpo_pair(dpo_file, prompt, response, ideal, category, score)
                    print(f"   ✓ Added to DPO training pairs")
                    result["dpo_saved"] = True

    print(f"\n   ✓ Scored: {score}/5")
    return result


# =============================================================================
# Main Runner Functions
# =============================================================================


async def run_golden_set_interactive(
    prompts_path: str = "tests/golden_set_prompts.json",
    output_dir: str = "tests/results",
    force_lang: Optional[Language] = None,
) -> Path:
    """
    Run Golden Set with interactive scoring after each response.
    """
    base_dir = Path(__file__).parent.parent
    prompts_file = base_dir / prompts_path if not Path(prompts_path).is_absolute() else Path(prompts_path)
    output_path = base_dir / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Golden Set prompts not found: {prompts_file}")

    with open(prompts_file) as f:
        golden_set = json.load(f)

    prompts: List[Dict[str, Any]] = golden_set.get("prompts", [])
    if not prompts:
        raise ValueError("No prompts found in Golden Set file")

    lang_mode = f"forced={force_lang}" if force_lang else "random (en/es)"

    print(f"\n{'═' * 70}")
    print(" GOLDEN SET EVALUATION - Interactive Mode")
    print(f"{'═' * 70}")
    print(f"Prompts to run: {len(prompts)}")
    print(f"Language mode: {lang_mode}")
    print(f"Output directory: {output_path}")
    print(f"\nThis mode will pause after each response for your scoring.")
    print("You can provide ideal responses for low scores (saved for DPO training).")
    print(f"{'═' * 70}\n")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # DPO training pairs file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dpo_file = output_path / f"dpo_pairs_{timestamp}.json"
    results_file = output_path / f"golden_set_results_{timestamp}.json"

    async with RAGService() as service:
        # Health check
        health = await service.health_check()
        print(f"Service status: {health['status']}")
        for component, info in health.get("components", {}).items():
            status = info.get("status", "unknown")
            print(f"  {component}: {status}")

        if health["status"] == "unhealthy":
            print(f"\n⚠ Service not fully healthy")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                raise RuntimeError("Aborted due to unhealthy service")

        input("\nPress Enter to start evaluation...")

        results: List[Dict[str, Any]] = []
        errors = 0
        lang_counts = {"en": 0, "es": 0}
        scores_collected = 0
        dpo_pairs_saved = 0

        for i, item in enumerate(prompts):
            prompt_id = item.get("id", f"prompt_{i+1}")
            category = item.get("category", "unknown")

            lang = select_language(force_lang)
            lang_counts[lang] += 1

            prompt_text = get_prompt_in_language(item, lang)
            original_prompt = item.get("prompt", "")

            lang_indicator = "🇪🇸" if lang == "es" else "🇺🇸"
            print(f"\n{'─' * 70}")
            print(f"Generating [{i+1}/{len(prompts)}] {lang_indicator} {category}...")

            try:
                response = await service.ask(prompt_text)

                result = {
                    "id": prompt_id,
                    "category": category,
                    "language": lang,
                    "prompt": prompt_text,
                    "prompt_original": original_prompt,
                    "expected_themes": item.get("key_themes", []),
                    "red_flags": item.get("red_flags", []),
                    "response": response.response,
                    "raw_response": response.raw_response,
                    "template_used": response.template_name,
                    "reasoning_stripped": response.reasoning_stripped,
                    "retrieval_time_ms": response.retrieval_time_ms,
                    "generation_time_ms": response.generation_time_ms,
                    "total_time_ms": response.total_time_ms,
                    "context_length": len(response.context_used),
                    "score": None,
                    "score_rationale": None,
                }

                # Interactive scoring
                result = await interactive_score(result, dpo_file, i + 1, len(prompts))

                if result.get("score") is not None:
                    scores_collected += 1
                if result.get("dpo_saved"):
                    dpo_pairs_saved += 1

                results.append(result)

            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted by user. Saving progress...")
                break

            except Exception as e:
                errors += 1
                results.append({
                    "id": prompt_id,
                    "category": category,
                    "language": lang,
                    "prompt": prompt_text,
                    "prompt_original": original_prompt,
                    "error": str(e),
                    "score": 1,
                    "score_rationale": f"Execution error: {e}",
                })
                print(f"  ✗ Error: {e}")

    # Save results
    output_data = {
        "metadata": {
            "run_timestamp": datetime.now().isoformat(),
            "mode": "interactive",
            "prompts_file": str(prompts_file),
            "total_prompts": len(prompts),
            "completed": len(results),
            "successful": len(results) - errors,
            "errors": errors,
            "scores_collected": scores_collected,
            "dpo_pairs_saved": dpo_pairs_saved,
            "language_mode": force_lang or "random",
            "language_distribution": lang_counts,
            "scoring_status": "complete" if scores_collected == len(results) else "partial",
        },
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'═' * 70}")
    print(" EVALUATION COMPLETE")
    print(f"{'═' * 70}")
    print(f"Completed: {len(results)}/{len(prompts)}")
    print(f"Errors: {errors}")
    print(f"Scores collected: {scores_collected}")
    print(f"DPO pairs saved: {dpo_pairs_saved}")
    print(f"Languages: 🇺🇸 {lang_counts['en']} English, 🇪🇸 {lang_counts['es']} Spanish")
    print(f"\nResults saved to: {results_file}")
    if dpo_pairs_saved > 0:
        print(f"DPO pairs saved to: {dpo_file}")
    print(f"\nRun scorer: python golden_set_scorer.py {results_file}")
    print(f"{'═' * 70}\n")

    return results_file


async def run_golden_set_batch(
    prompts_path: str = "tests/golden_set_prompts.json",
    output_dir: str = "tests/results",
    force_lang: Optional[Language] = None,
) -> Path:
    """
    Run all Golden Set prompts in batch mode (score later).
    """
    base_dir = Path(__file__).parent.parent
    prompts_file = base_dir / prompts_path if not Path(prompts_path).is_absolute() else Path(prompts_path)
    output_path = base_dir / output_dir if not Path(output_dir).is_absolute() else Path(output_path)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Golden Set prompts not found: {prompts_file}")

    with open(prompts_file) as f:
        golden_set = json.load(f)

    prompts: List[Dict[str, Any]] = golden_set.get("prompts", [])
    if not prompts:
        raise ValueError("No prompts found in Golden Set file")

    lang_mode = f"forced={force_lang}" if force_lang else "random (en/es)"

    print(f"\n{'=' * 60}")
    print(" GOLDEN SET EVALUATION - Batch Mode")
    print(f"{'=' * 60}")
    print(f"Prompts to run: {len(prompts)}")
    print(f"Language mode: {lang_mode}")
    print(f"Output directory: {output_path}\n")

    async with RAGService() as service:
        health = await service.health_check()
        print(f"Service status: {health['status']}")
        for component, info in health.get("components", {}).items():
            status = info.get("status", "unknown")
            print(f"  {component}: {status}")

        if health["status"] == "unhealthy":
            print(f"\n⚠ Service not fully healthy")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                raise RuntimeError("Aborted due to unhealthy service")

        print(f"\n{'─' * 60}")
        print(" Running prompts...")
        print(f"{'─' * 60}\n")

        results: List[Dict[str, Any]] = []
        errors = 0
        lang_counts = {"en": 0, "es": 0}

        for i, item in enumerate(prompts):
            prompt_id = item.get("id", f"prompt_{i+1}")
            category = item.get("category", "unknown")

            lang = select_language(force_lang)
            lang_counts[lang] += 1

            prompt_text = get_prompt_in_language(item, lang)
            original_prompt = item.get("prompt", "")

            lang_indicator = "🇪🇸" if lang == "es" else "🇺🇸"
            display_text = prompt_text[:45] + "..." if len(prompt_text) > 45 else prompt_text
            print(f"[{i+1:2}/{len(prompts)}] {lang_indicator} {category}: {display_text}")

            try:
                response = await service.ask(prompt_text)

                results.append({
                    "id": prompt_id,
                    "category": category,
                    "language": lang,
                    "prompt": prompt_text,
                    "prompt_original": original_prompt,
                    "expected_themes": item.get("key_themes", []),
                    "red_flags": item.get("red_flags", []),
                    "response": response.response,
                    "raw_response": response.raw_response,
                    "template_used": response.template_name,
                    "reasoning_stripped": response.reasoning_stripped,
                    "retrieval_time_ms": response.retrieval_time_ms,
                    "generation_time_ms": response.generation_time_ms,
                    "total_time_ms": response.total_time_ms,
                    "context_length": len(response.context_used),
                    "score": None,
                    "score_rationale": None,
                })
                print(f"         ✓ Generated ({response.generation_time_ms:.0f}ms)")

            except Exception as e:
                errors += 1
                results.append({
                    "id": prompt_id,
                    "category": category,
                    "language": lang,
                    "prompt": prompt_text,
                    "prompt_original": original_prompt,
                    "error": str(e),
                    "score": 1,
                    "score_rationale": f"Execution error: {e}",
                })
                print(f"         ✗ Error: {e}")

    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"golden_set_results_{timestamp}.json"

    output_data = {
        "metadata": {
            "run_timestamp": datetime.now().isoformat(),
            "mode": "batch",
            "prompts_file": str(prompts_file),
            "total_prompts": len(prompts),
            "successful": len(prompts) - errors,
            "errors": errors,
            "language_mode": force_lang or "random",
            "language_distribution": lang_counts,
            "scoring_status": "pending",
        },
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(" RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successful: {len(prompts) - errors}/{len(prompts)}")
    print(f"Errors: {errors}")
    print(f"Languages: 🇺🇸 {lang_counts['en']} English, 🇪🇸 {lang_counts['es']} Spanish")
    print(f"\nResults saved to: {results_file}")
    print(f"\n{'─' * 60}")
    print(" NEXT STEPS:")
    print("─" * 60)
    print("1. Open the results JSON file")
    print("2. For each response, add:")
    print("   - 'score': 1-5 (see rubric)")
    print("   - 'score_rationale': Brief explanation")
    print("3. Run golden_set_scorer.py to calculate metrics")
    print(f"{'=' * 60}\n")

    return results_file


async def run_single_prompt(prompt: str) -> None:
    """Run a single prompt for quick testing."""
    print(f"\n{'=' * 60}")
    print(" SINGLE PROMPT TEST")
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt}\n")

    async with RAGService() as service:
        health = await service.health_check()
        print(f"Service status: {health['status']}")

        if health["status"] == "unhealthy":
            print("⚠ Service unhealthy - aborting")
            return

        print("\nGenerating response...\n")
        response = await service.ask(prompt)

        print(f"{'─' * 60}")
        print(f"Template: {response.template_name}")
        print(f"Reasoning stripped: {response.reasoning_stripped}")
        print(f"Context length: {len(response.context_used)} chars")
        print(f"Retrieval time: {response.retrieval_time_ms:.0f}ms")
        print(f"Generation time: {response.generation_time_ms:.0f}ms")
        print(f"Total time: {response.total_time_ms:.0f}ms")
        print(f"{'─' * 60}")
        print("\nRESPONSE:\n")
        print(response.response)
        print(f"\n{'=' * 60}\n")


def parse_args() -> tuple:
    """Parse command line arguments."""
    prompts_path = "tests/golden_set_prompts.json"
    force_lang: Optional[Language] = None
    single_prompt: Optional[str] = None
    batch_mode = False  # Interactive is default

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--batch" or arg == "-b":
            batch_mode = True
            i += 1
        elif arg == "--interactive" or arg == "-i":
            batch_mode = False
            i += 1
        elif arg == "--lang" and i + 1 < len(args):
            lang = args[i + 1].lower()
            if lang in ("en", "es"):
                force_lang = lang  # type: ignore
            else:
                print(f"Invalid language: {lang}. Use 'en' or 'es'.")
                sys.exit(1)
            i += 2
        elif arg == "--single":
            if i + 1 < len(args):
                single_prompt = " ".join(args[i + 1:])
            else:
                single_prompt = input("Enter prompt: ")
            break
        elif arg.endswith(".json") or Path(arg).exists():
            prompts_path = arg
            i += 1
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            single_prompt = " ".join(args[i:])
            break

    return prompts_path, force_lang, single_prompt, batch_mode


if __name__ == "__main__":
    prompts_path, force_lang, single_prompt, batch_mode = parse_args()

    if single_prompt:
        asyncio.run(run_single_prompt(single_prompt))
    elif batch_mode:
        asyncio.run(run_golden_set_batch(prompts_path, force_lang=force_lang))
    else:
        asyncio.run(run_golden_set_interactive(prompts_path, force_lang=force_lang))
