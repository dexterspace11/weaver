Knowledge Weaver Turn scattered research texts into one deep, emergent insight – locally & privately.Knowledge Weaver is a beautiful Streamlit app that helps researchers, students, and anyone working with multiple papers or field notes. It extracts key findings from each text and then weaves them together into a single, original synthesis that reveals connections you might otherwise miss.Perfect for literature reviews, meta-analysis prep, grant proposals, or simply making sense of your reading notes.FeaturesPaste any number of study abstracts, excerpts, or full texts (2–10 recommended)
Automatic high-quality summaries (powered by BART)
One-click generation of a true emergent insight that combines all inputs
Fully local & private – no data ever leaves your computer
Runs great even on modest laptops (Core i3 + 8 GB RAM)
Export everything as a clean Markdown report

How It WorksInput Studies – Paste your texts.
Extract Key Findings – Get concise summaries for each one.
Central Question – Write the connecting theme (e.g., "traditional medicinal plants in protected forests").
Generate Woven Insight – Watch Phi-3 Mini (or another local model) create a deep, original synthesis.

Example OutputInputs  Ethnobotanical documentation of 261 angiosperm species used for skin ailments in Bhandara district, India  
Firewood species selection and cultural importance in rural Limpopo Province, South Africa

Emergent Insight  It becomes evident from synthesizing ethnobotanical knowledge across diverse cultures and regions… that there exists a rich tapestry of botanic utility deeply intertwined with local traditions… underscoring a resilient symbiosis between biodiversity and cultural identity.
RequirementsWindows / macOS / Linux
Python 3.10–3.12
Ollama (provides the powerful local LLM)

Quick Start (Windows)Install Python from https://python.org
Install Ollama from https://ollama.com (one-click installer)
Open Command Prompt and download the model:

ollama run phi3:mini

(type exit when it finishes)
Install Python packages:

pip install streamlit ollama transformers torch

Save the script as weaver.py (or any name)
Run:

streamlit run weaver.py

Your browser will open the app automatically at http://localhost:8501Optional ModelsIn the sidebar you can switch to:gemma2:2b – slightly faster
llama3.2:1b or llama3.2:3b – different styles

All run locally via Ollama.LicenseMIT – feel free to use, modify, and share.Enjoy weaving knowledge! Created with  by a researcher who wanted better synthesis tools. Happy exploring!

Add troubleshooting section

Deploy to Streamlit Cloud

Make it more concise

