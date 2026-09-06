#!/usr/bin/env python3
"""
Autonomous AI Research Agent
Combines 6 AI models for research automation
"""

import os
import json
from pathlib import Path
from datetime import datetime
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

if HF_TOKEN:
    login(token=HF_TOKEN)

class AutonomousAgent:
    def __init__(self):
        print("🚀 Initializing Autonomous Agent...")
        
        # Model 1: Text generation (planning)
        print("  [1/6] Loading text generator...")
        self.text_gen = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",  # Start small, upgrade later
            device_map="auto",
            max_new_tokens=512
        )
        
        # Model 2: Code generation
        print("  [2/6] Loading code generator...")
        self.code_gen = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            device_map="auto",
            max_new_tokens=512
        )
        
        # Model 3: Image captioning
        self.captioner = pipeline(
            task="image-text-to-text",
            model="Salesforce/blip2-opt-2.7b",
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Model 4: Web search
        print("  [4/6] Setting up web search...")
        from duckduckgo_search import DDGS
        self.search = DDGS()
        
        # Model 5 & 6: Will add later (vision + audio)
        print("  [5/6] Vision model: Ready to add")
        print("  [6/6] Audio model: Ready to add")
        
        print("✅ Agent initialized!\n")
    
    def search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web"""
        print(f"🔍 Searching: {query}")
        try:
            results = self.search.text(query, max_results=max_results)
            formatted = "\n".join([
                f"- {r['title']}: {r['body']}" 
                for r in results
            ])
            return formatted
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def generate_text(self, prompt: str) -> str:
        """Generate text response"""
        print("💭 Generating response...")
        result = self.text_gen(prompt, max_new_tokens=512)
        return result[0]["generated_text"]
    
    def generate_code(self, task: str) -> str:
        """Generate Python code"""
        print("💻 Generating code...")
        prompt = f"""Write Python code for this task:
{task}

Include:
- Imports
- Comments
- Error handling
- Example usage

Code:"""
        result = self.code_gen(prompt, max_new_tokens=512)
        return result[0]["generated_text"]
    
    def analyze_image(self, image_path: str) -> str:
        """Analyze an image"""
        print(f"🖼️  Analyzing: {image_path}")
        image = Image.open(image_path)
        result = self.captioner(image)
        return result[0]["generated_text"]
    
    def research_task(self, topic: str) -> dict:
        """Complete research workflow"""
        print("\n" + "="*60)
        print(f"🎯 RESEARCH TASK: {topic}")
        print("="*60)
        
        results = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }
        
        # Step 1: Web search
        print("\n[Step 1/3] Searching web...")
        search_results = self.search_web(f"{topic} 2025 2026")
        results["web_search"] = search_results
        results["steps"].append("Web search completed")
        
        # Step 2: Generate analysis
        print("\n[Step 2/3] Analyzing information...")
        analysis_prompt = f"""Based on this research about {topic}, provide a comprehensive analysis:

{search_results[:2000]}

Include:
1. Key findings
2. Recent developments
3. Practical applications
4. Future trends

Analysis:"""
        analysis = self.generate_text(analysis_prompt)
        results["analysis"] = analysis
        results["steps"].append("Analysis completed")
        
        # Step 3: Generate code example
        print("\n[Step 3/3] Generating code example...")
        code = self.generate_code(f"Create a Python script related to: {topic}")
        results["code_example"] = code
        results["steps"].append("Code generation completed")
        
        # Save results
        output_dir = Path("agent_output")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"research_{topic.replace(' ', '_')[:30]}.json"
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Research complete! Saved to: {output_file}")
        print("="*60 + "\n")
        
        return results


def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════╗
║                                                  ║
║      AUTONOMOUS AI RESEARCH AGENT v0.1           ║
║                                                  ║
╚══════════════════════════════════════════════════╝
    """)
    
    # Initialize agent
    agent = AutonomousAgent()
    
    # Example research task
    topic = "Latest advancements in transformer architectures 2026"
    
    print(f"\n💡 Running example research on: {topic}\n")
    
    # Run research
    results = agent.research_task(topic)
    
    # Print summary
    print("📊 RESULTS SUMMARY:")
    print(f"  Topic: {results['topic']}")
    print(f"  Steps completed: {len(results['steps'])}")
    print(f"  Output file: agent_output/research_*.json")
    print("\n✅ Done! Check agent_output/ folder for results.")


if __name__ == "__main__":
    main()
