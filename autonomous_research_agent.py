%%writefile autonomous_research_agent.py
#!/usr/bin/env python3
"""
GHOSTWORKER AI v1.0
A multi-model autonomous assistant for planning, coding,
image understanding, audio transcription, and final reporting.

Models:
1. Planner Agent: Qwen/Qwen2.5-1.5B-Instruct
2. Coding Agent: Qwen/Qwen2.5-Coder-1.5B-Instruct
3. Vision Agent: Salesforce/blip-image-captioning-base
4. Audio Agent: openai/whisper-base (loaded only when required)

Designed for Kaggle Tesla T4 GPU environments.
"""

import os
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

load_dotenv()

PROJECT_DIR = Path("agent_output")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

PLANNER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CODER_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
VISION_MODEL = "Salesforce/blip-image-captioning-base"
AUDIO_MODEL = "openai/whisper-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------------
# MEMORY MANAGEMENT
# ---------------------------------------------------------------------

def clear_gpu_memory():
    """Release unused RAM and GPU VRAM before loading another model."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_status():
    """Print basic GPU memory information."""
    if not torch.cuda.is_available():
        print("ℹ️ Running on CPU.")
        return

    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(index) / (1024 ** 3)
        name = torch.cuda.get_device_name(index)

        print(
            f"GPU {index}: {name} | "
            f"Allocated: {allocated:.2f} GB | "
            f"Reserved: {reserved:.2f} GB"
        )


# ---------------------------------------------------------------------
# MAIN MULTI-MODEL AGENT
# ---------------------------------------------------------------------

class GhostWorkerAI:
    """
    A multi-model AI system.

    Important design:
    - Models are loaded only when needed.
    - A model is removed from memory after it completes its task.
    - This makes the application more reliable on free T4 GPUs.
    """

    def __init__(self):
        print("\n" + "=" * 68)
        print("🤖 GHOSTWORKER AI — MULTI-MODEL AUTONOMOUS ASSISTANT")
        print("=" * 68)
        print(f"Device: {DEVICE}")
        print(f"PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print_gpu_status()

        print("\n✅ System initialized.")
        print("✅ Models will load one at a time to protect GPU memory.\n")

    # -----------------------------------------------------------------
    # MODEL 1: PLANNER AGENT
    # -----------------------------------------------------------------

    def create_plan(self, user_goal: str) -> str:
        """Use Qwen Instruct to turn a large goal into an execution plan."""
        print("\n🧠 [MODEL 1/4] Loading Planner Agent...")

        planner = pipeline(
            task="text-generation",
            model=PLANNER_MODEL,
            device_map="auto",
            torch_dtype=DTYPE,
        )

        prompt = f"""You are the Master Planner in a multi-model autonomous AI system.

The user wants to achieve this goal:

{user_goal}

Create a clear, realistic, step-by-step execution plan.

Use exactly this structure:

GOAL:
Write the main goal in one sentence.

SUBTASKS:
1. [Research Agent] ...
2. [Coding Agent] ...
3. [Vision Agent] ...
4. [Audio Agent] ...
5. [Quality Agent] ...

EXPECTED FINAL OUTPUT:
...

RISKS AND SOLUTIONS:
- Risk: ...
  Solution: ...

Keep the plan practical for a single developer using free cloud resources.
"""

        response = planner(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.4,
            return_full_text=False,
        )

        result = response[0]["generated_text"].strip()

        del planner
        clear_gpu_memory()

        print("✅ Planner Agent completed the task.")
        return result

    # -----------------------------------------------------------------
    # MODEL 2: CODING AGENT
    # -----------------------------------------------------------------

    def generate_python_code(self, task: str) -> str:
        """Use Qwen Coder to generate a complete Python solution."""
        print("\n💻 [MODEL 2/4] Loading Coding Agent...")

        coder = pipeline(
            task="text-generation",
            model=CODER_MODEL,
            device_map="auto",
            torch_dtype=DTYPE,
        )

        prompt = f"""You are the Coding Agent in a multi-model AI project.

Write a complete and executable Python program for this task:

{task}

Requirements:
- Return only Python code, without Markdown fences.
- Include all imports.
- Use clear function names.
- Add concise comments.
- Include safe error handling.
- Add a main() function.
- Add: if __name__ == "__main__": main()
- Avoid paid APIs.
- Prefer standard Python libraries when possible.
"""

        response = coder(
            prompt,
            max_new_tokens=700,
            do_sample=True,
            temperature=0.2,
            return_full_text=False,
        )

        result = response[0]["generated_text"].strip()

        # Remove accidental Markdown code fences if generated.
        result = result.replace("```python", "").replace("```", "").strip()

        del coder
        clear_gpu_memory()

        print("✅ Coding Agent completed the task.")
        return result

    # -----------------------------------------------------------------
    # MODEL 3: VISION AGENT
    # -----------------------------------------------------------------

    def analyze_image(self, image_path: str, instruction: str = "") -> str:
        """
        Use BLIP for image captioning / visual analysis.
        BLIP is loaded using its official Processor + Model interface,
        rather than the unsupported image-to-text pipeline.
        """
        print("\n🖼️ [MODEL 3/4] Loading Vision Agent...")

        path = Path(image_path)

        if not path.exists():
            return f"❌ Image not found: {path}"

        try:
            image = Image.open(path).convert("RGB")
        except Exception as error:
            return f"❌ Could not open image: {error}"

        processor = BlipProcessor.from_pretrained(VISION_MODEL)

        model = BlipForConditionalGeneration.from_pretrained(
            VISION_MODEL,
            torch_dtype=DTYPE,
        ).to(DEVICE)

        if instruction.strip():
            prompt = instruction.strip()
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(DEVICE, DTYPE)
        else:
            inputs = processor(
                images=image,
                return_tensors="pt",
            ).to(DEVICE, DTYPE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
            )

        result = processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        del model
        del processor
        clear_gpu_memory()

        print("✅ Vision Agent completed the task.")
        return result

    # -----------------------------------------------------------------
    # MODEL 4: AUDIO AGENT
    # -----------------------------------------------------------------

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Use Whisper Base to transcribe audio.
        The model is only downloaded and loaded when this method is used.
        """
        print("\n🎙️ [MODEL 4/4] Loading Audio Agent...")

        path = Path(audio_path)

        if not path.exists():
            return f"❌ Audio file not found: {path}"

        transcriber = pipeline(
            task="automatic-speech-recognition",
            model=AUDIO_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=DTYPE,
        )

        try:
            response = transcriber(
                str(path),
                chunk_length_s=30,
                return_timestamps=True,
            )

            result = response["text"].strip()

        except Exception as error:
            result = f"❌ Audio transcription failed: {error}"

        del transcriber
        clear_gpu_memory()

        print("✅ Audio Agent completed the task.")
        return result

    # -----------------------------------------------------------------
    # FINAL REPORT — USES PLANNER MODEL AGAIN
    # -----------------------------------------------------------------

    def create_final_report(
        self,
        goal: str,
        plan: str,
        code: str,
        image_analysis: str = "",
        audio_transcript: str = "",
    ) -> str:
        """Use the Planner model again to form a concise final report."""
        print("\n📝 Creating final report with the Planner Agent...")

        planner = pipeline(
            task="text-generation",
            model=PLANNER_MODEL,
            device_map="auto",
            torch_dtype=DTYPE,
        )

        prompt = f"""You are the Report Agent in a multi-model AI system.

Create a concise project report from the following information.

USER GOAL:
{goal}

EXECUTION PLAN:
{plan[:2500]}

CODE AGENT OUTPUT:
{code[:2500]}

VISION AGENT OUTPUT:
{image_analysis[:1000] if image_analysis else "No image was supplied."}

AUDIO AGENT OUTPUT:
{audio_transcript[:1000] if audio_transcript else "No audio was supplied."}

Use this exact Markdown structure:

# Project Report

## Objective

## Multi-Model Architecture

## Workflow Completed

## Key Output

## Limitations

## Next Improvements
"""

        response = planner(
            prompt,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.3,
            return_full_text=False,
        )

        result = response[0]["generated_text"].strip()

        del planner
        clear_gpu_memory()

        return result

    # -----------------------------------------------------------------
    # FULL WORKFLOW
    # -----------------------------------------------------------------

    def run_full_workflow(
        self,
        goal: str,
        code_task: str,
        image_path: str = "",
        audio_path: str = "",
    ) -> dict:
        """Run a complete multi-model workflow and save all outputs."""

        print("\n" + "=" * 68)
        print("🚀 STARTING FULL AUTONOMOUS MULTI-MODEL WORKFLOW")
        print("=" * 68)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "timestamp": timestamp,
            "goal": goal,
            "models_used": [
                PLANNER_MODEL,
                CODER_MODEL,
                VISION_MODEL,
            ],
        }

        # 1. Planning
        plan = self.create_plan(goal)
        results["plan"] = plan

        # 2. Code generation
        code = self.generate_python_code(code_task)
        results["generated_code"] = code

        # 3. Optional image analysis
        image_analysis = ""

        if image_path:
            image_analysis = self.analyze_image(
                image_path=image_path,
                instruction="a detailed description of",
            )
            results["image_analysis"] = image_analysis
            results["models_used"].append(VISION_MODEL)

        # 4. Optional audio transcription
        audio_transcript = ""

        if audio_path:
            audio_transcript = self.transcribe_audio(audio_path)
            results["audio_transcript"] = audio_transcript
            results["models_used"].append(AUDIO_MODEL)

        # 5. Final report
        report = self.create_final_report(
            goal=goal,
            plan=plan,
            code=code,
            image_analysis=image_analysis,
            audio_transcript=audio_transcript,
        )
        results["final_report"] = report

        # Save all project outputs.
        json_file = PROJECT_DIR / f"workflow_{timestamp}.json"
        report_file = PROJECT_DIR / f"report_{timestamp}.md"
        code_file = PROJECT_DIR / f"generated_code_{timestamp}.py"

        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        report_file.write_text(report, encoding="utf-8")
        code_file.write_text(code, encoding="utf-8")

        print("\n" + "=" * 68)
        print("✅ WORKFLOW FINISHED SUCCESSFULLY")
        print("=" * 68)
        print(f"📁 JSON results: {json_file}")
        print(f"📄 Report:       {report_file}")
        print(f"💻 Generated code: {code_file}")
        print(f"🤖 Models used: {len(set(results['models_used']))}")
        print("=" * 68 + "\n")

        return results


# ---------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GhostWorker AI — Multi-Model Autonomous Assistant"
    )

    parser.add_argument(
        "--goal",
        type=str,
        default=(
            "Design an autonomous multi-model AI assistant that helps "
            "students turn an idea into a research plan, Python code, "
            "image insight, and final report."
        ),
        help="Main user goal for the planner agent."
    )

    parser.add_argument(
        "--code-task",
        type=str,
        default=(
            "Build a Python command-line program that reads a CSV file, "
            "calculates basic statistics for numeric columns, and saves "
            "the results as a JSON file."
        ),
        help="Task sent to the Coding Agent."
    )

    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional local image path for vision analysis."
    )

    parser.add_argument(
        "--audio",
        type=str,
        default="",
        help="Optional local audio path for Whisper transcription."
    )

    args = parser.parse_args()

    agent = GhostWorkerAI()

    results = agent.run_full_workflow(
        goal=args.goal,
        code_task=args.code_task,
        image_path=args.image,
        audio_path=args.audio,
    )

    print("FINAL REPORT PREVIEW:\n")
    print(results["final_report"])


if __name__ == "__main__":
    main()
