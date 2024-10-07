# 💾 LLM Datasets: Unlocking the Potential of Large Language Models

🤗 [Hugging Face](https://huggingface.co/llmat) • 💻 [Blog](https://mattdepaolis.github.io/blog/)  
_Top-quality datasets, tools, and ideas for enhancing Large Language Models (LLMs)._

## 📑 Table of Contents

- [💾 LLM Datasets: Unlocking the Potential of Large Language Models](#-llm-datasets-unlocking-the-potential-of-large-language-models)
  - [📑 Table of Contents](#-table-of-contents)
  - [Introduction](#introduction)
  - [🌟 The Essence of a Great Dataset](#-the-essence-of-a-great-dataset)
  - [📚 Open-Source Datasets](#-open-source-datasets)
  - [⚙️ Pre-Training Datasets](#️-pre-training-datasets)
  - [🛠️ Supervised Fine-Tuning Datasets](#️-supervised-fine-tuning-datasets)
    - [General-Purpose Datasets](#general-purpose-datasets)
    - [🧮 Math \& Logic](#-math--logic)
    - [💻 Code](#-code)
    - [🗣️ Conversation \& Role-Play](#️-conversation--role-play)
    - [🤖 Agent \& Function Calling](#-agent--function-calling)
  - [⚖️ Preference Alignment Datasets](#️-preference-alignment-datasets)
  - [🛠️ Tools for Creating High-Quality Datasets](#️-tools-for-creating-high-quality-datasets)
    - [🧹 Data Deduplication and Cleaning](#-data-deduplication-and-cleaning)
    - [✅ Evaluating Data Quality](#-evaluating-data-quality)
    - [🛠️ Generating Additional Data](#️-generating-additional-data)
      - [Supervised Fine-Tuning (SFT) Datasets](#supervised-fine-tuning-sft-datasets)
      - [Pre-Training Datasets](#pre-training-datasets)
    - [🔍 Exploring and Visualizing Data](#-exploring-and-visualizing-data)
    - [🌐 Data Scraping](#-data-scraping)
  - [Conclusion](#conclusion)

## Introduction

Welcome to your ultimate resource for enhancing Large Language Models (LLMs) through top-quality datasets, cutting-edge tools, and innovative ideas. Whether you’re building a model from scratch or fine-tuning an existing one, the data you use is crucial. This guide will walk you through what makes a great dataset, provide curated lists of open-source datasets for various training stages, and introduce tools to help you create and manage high-quality data effectively.

---

## 🌟 The Essence of a Great Dataset

A high-quality dataset is the backbone of any successful LLM. But what exactly makes a dataset exceptional? Here are the key attributes:

**• Accuracy:** Information should be correct, relevant, and clearly articulated. Responses must directly address the given questions or instructions.

**• Diversity:** A wide range of topics, styles, and contexts ensures the model can handle different tasks and follow diverse instructions effectively.

**• Complexity:** Including challenging tasks that require multi-step reasoning or problem-solving helps the model manage more intricate queries.

Evaluating these aspects can be tricky. For example, checking accuracy is straightforward for math problems but less so for open-ended questions. Diversity can be measured by the range of topics covered, and complexity can be assessed using other language models as evaluators.

---

## 📚 Open-Source Datasets

## ⚙️ Pre-Training Datasets

Pre-training datasets provide the foundational understanding of language, context, and general knowledge that LLMs need. They enable models to learn useful representations and patterns that can be fine-tuned for various downstream tasks.

| **Dataset**                                                              | **Size** | **Authors** | **Date**    | **Description**                                                                                                                                                                                                           |
| ------------------------------------------------------------------------ | -------- | ----------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)         | 46B      | HuggingFace | July 2024   | The 🍷 FineWeb dataset consists of more than 15T tokens of cleaned and deduplicated english web data from CommonCrawl. The data processing pipeline is optimized for LLM performance and ran on the 🏭 datatrove library. |
| [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 3B       | HuggingFace | August 2024 | 📚 FineWeb-Edu dataset consists of 1.3T tokens and 5.4T tokens (FineWeb-Edu-score-2) of educational web pages filtered from 🍷 FineWeb dataset. This is the 1.3 trillion version.                                         |

---

## 🛠️ Supervised Fine-Tuning Datasets

After initial training, fine-tuning with specialized datasets transforms an LLM into a versatile assistant capable of answering questions and performing various tasks. These datasets consist of instruction-response pairs and are available under permissive licenses.

### General-Purpose Datasets

Designed to make models versatile by exposing them to a broad spectrum of high-quality data, these datasets often combine real-world information with synthetic data generated by advanced models like GPT-4.

| **Dataset**                                                                                                   | **Size** | **Authors**                  | **Date** | **Description**                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Buzz](https://huggingface.co/datasets/H-D-T/Buzz-slice-1-10-V1.2)                                            | 31.2M    | Alignment Lab AI             | May 2024 | Extensive collection using data augmentation and deduplication techniques.                                                                                                    |
| [WebInstructSub](https://huggingface.co/datasets/chargoddard/WebInstructSub-prometheus)                       | 2.39M    | Yue et al.                   | May 2024 | Derived from Common Crawl documents, extracting and refining QA pairs. [MAmmoTH2 paper](https://arxiv.org/abs/2405.03548) (subset).                                           |
| [The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)                                                 | 1.75M    | Arcee AI                     | Jul 2024 | Filtered for instruction following. [100k subset](https://huggingface.co/datasets/mlabonne/FineTome-100k).                                                                    |
| [Hercules v4.5](https://huggingface.co/datasets/Locutusque/hercules-v4.5)                                     | 1.72M    | Sebastian Gabarain           | Apr 2024 | Covers math, code, role-playing, etc. [v4](https://huggingface.co/datasets/Locutusque/hercules-v4.0) for more details.                                                        |
| [Dolphin-2.9](https://huggingface.co/datasets/cognitivecomputations/Dolphin-2.9)                              | 1.39M    | Cognitive Computations       | Apr 2023 | Large-scale general-purpose dataset for Dolphin models.                                                                                                                       |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)                                            | 1.04M    | Zhao et al.                  | May 2023 | Real conversations with GPT-3.5/4, including metadata. [WildChat paper](https://arxiv.org/abs/2405.01470).                                                                    |
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)                                      | 1M       | Teknium                      | Nov 2023 | Large-scale dataset for OpenHermes models.                                                                                                                                    |
| [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)                                   | 660k     | BAAI                         | Jun 2024 | Based on a curated collection of evolved instructions.                                                                                                                        |
| [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca)                                                | 518k     | Lian et al.                  | Sep 2023 | Curated subset of [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) using GPT-4 to eliminate incorrect answers.                                                  |
| [Tulu V2 Mix](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture)                                    | 326k     | Ivison et al.                | Nov 2023 | Mix of high-quality datasets. [Tulu 2 paper](https://arxiv.org/abs/2311.10702).                                                                                               |
| [UltraInteract SFT](https://huggingface.co/datasets/openbmb/UltraInteract_sft)                                | 289k     | Yuan et al.                  | Apr 2024 | Focused on math, coding, and logic with step-by-step answers. [Eurus paper](https://arxiv.org/abs/2404.02078).                                                                |
| [NeurIPS-LLM-data](https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data)                                  | 204k     | Jindal et al.                | Nov 2023 | Winner of the [NeurIPS LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io/).                                                                                |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                                | 200k     | Tunstall et al., Ding et al. | Oct 2023 | Filtered version of [UltraChat](https://github.com/thunlp/UltraChat) with 1.4M ChatGPT-generated dialogues.                                                                   |
| [WizardLM_evol_instruct_V2](https://huggingface.co/datasets/mlabonne/WizardLM_evol_instruct_v2_196K-ShareGPT) | 143k     | Xu et al.                    | Jun 2023 | Latest Evol-Instruct version applied to Alpaca and ShareGPT data. [WizardLM paper](https://arxiv.org/abs/2304.12244).                                                         |
| [Synthia-v1.3](https://huggingface.co/datasets/migtissera/Synthia-v1.3)                                       | 119k     | Migel Tissera                | Nov 2023 | High-quality synthetic data generated with GPT-4.                                                                                                                             |
| [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)                                                | 84.4k    | Köpf et al.                  | Mar 2023 | Human-generated assistant conversations in 35 languages. [OASST1 paper](https://arxiv.org/abs/2304.07327) and [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2). |
| [WizardLM_evol_instruct_70k](https://huggingface.co/datasets/mlabonne/WizardLM_evol_instruct_70k-ShareGPT)    | 70k      | Xu et al.                    | Apr 2023 | Evol-Instruct applied to Alpaca and ShareGPT. [WizardLM paper](https://arxiv.org/abs/2304.12244).                                                                             |
| [airoboros-3.2](https://huggingface.co/datasets/jondurbin/airoboros-3.2)                                      | 58.7k    | Jon Durbin                   | Dec 2023 | High-quality uncensored dataset.                                                                                                                                              |
| [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)       | 53k      | anon8231489123               | Mar 2023 | Filtered ShareGPT dataset with real user-ChatGPT conversations.                                                                                                               |
| [lmsys-chat-1m-smortmodelsonly](https://huggingface.co/datasets/Nebulous/lmsys-chat-1m-smortmodelsonly)       | 45.8k    | Nebulous, Zheng et al.       | Sep 2023 | Filtered [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) with responses from multiple models.                                                            |
| [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)                                   | 24.9k    | Lee et al.                   | Sep 2023 | Deduplicated datasets using Sentence Transformers, includes an NC dataset. [Platypus paper](https://arxiv.org/abs/2308.07317).                                                |
| [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)                       | 15k      | Conover et al.               | May 2023 | Created by Databricks employees with prompt-response pairs across eight instruction categories.                                                                               |

### 🧮 Math & Logic

LLMs often find mathematical reasoning and formal logic challenging. Specialized datasets help improve these areas by providing problems that require systematic thinking and multi-step reasoning.

| **Dataset**                                                                         | **Size** | **Authors**      | **Date** | **Description**                                                                                                                           |
| ----------------------------------------------------------------------------------- | -------- | ---------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)     | 5.75M    | Toshniwal et al. | Feb 2024 | Includes math problems from GSM8K and MATH with solutions from Mixtral-8x7B.                                                              |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)                  | 395k     | Yu et al.        | Dec 2023 | Mathematical questions rewritten from multiple perspectives for deeper understanding. [MetaMath paper](https://arxiv.org/abs/2309.12284). |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)              | 262k     | Yue et al.       | Sep 2023 | Compiled from 13 math datasets, focusing on chain-of-thought and program-of-thought reasoning.                                            |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200k     | Mitra et al.     | Feb 2024 | Grade school math problems generated using GPT-4 Turbo. [Orca-Math paper](https://arxiv.org/pdf/2402.14830.pdf).                          |

### 💻 Code

Enhancing coding capabilities in LLMs requires specialized datasets filled with diverse programming examples and challenges.

| **Dataset**                                                                                                            | **Size** | **Authors**    | **Date** | **Description**                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------- | -------- | -------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)           | 157k     | Zheng et al.   | Feb 2024 | Filtered version combining Magicoder-OSS-Instruct and other datasets to ensure high code quality.                                                                                                              |
| [Tested-143k-Python-Alpaca](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca)                          | 143k     | Vezora         | Mar 2024 | Python code that has passed automated tests for accuracy.                                                                                                                                                      |
| [glaive-code-assistant](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)                                | 136k     | Glaive.ai      | Sep 2023 | Synthetic problems and solutions with about 60% Python content. [v2](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v2) available.                                                             |
| [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)                  | 110k     | Wei et al.     | Nov 2023 | Cleaned version of [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) following StarCoder's decontamination process. [Magicoder paper](https://arxiv.org/abs/2312.02120). |
| [dolphin-coder](https://huggingface.co/datasets/cognitivecomputations/dolphin-coder)                                   | 109k     | Eric Hartford  | Nov 2023 | Transformed from [leetcode-rosetta](https://www.kaggle.com/datasets/erichartford/leetcode-rosetta).                                                                                                            |
| [synthetic_tex_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)                                 | 100k     | Gretel.ai      | Apr 2024 | Synthetic text-to-SQL samples covering various domains.                                                                                                                                                        |
| [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)                                         | 78.6k    | b-mc2          | Apr 2023 | Enhanced version of [WikiSQL](https://huggingface.co/datasets/wikisql) and [Spider](https://huggingface.co/datasets/spider).                                                                                   |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)                      | 75k      | Wei et al.     | Nov 2023 | Generated by `gpt-3.5-turbo-1106`. [Magicoder paper](https://arxiv.org/abs/2312.02120).                                                                                                                        |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)                                                   | 66.4k    | Zheng et al.   | Feb 2024 | Diverse Code Interpreter-like dataset with multi-turn dialogues and mixed text-code responses. [OpenCodeInterpreter paper](https://arxiv.org/abs/2402.14658).                                                  |
| [Open-Critic-GPT](https://huggingface.co/datasets/Vezora/Open-Critic-GPT)                                              | 55.1k    | Vezora         | Jul 2024 | Uses a local model to create and identify bugs in code across various programming languages.                                                                                                                   |
| [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | 50.7k    | Lozhkov et al. | Apr 2024 | Created using seed functions from TheStack v1, self-instruction with StarCoder2, and self-validation. [Blog post](https://huggingface.co/blog/sc2-instruct).                                                   |

### 🗣️ Conversation & Role-Play

To excel in conversational settings, LLMs benefit from datasets that mimic real-life dialogues and role-playing scenarios.

| **Dataset**                                                                                              | **Size** | **Authors**             | **Date** | **Description**                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------- | -------- | ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Bluemoon](https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned)                      | 290k     | Squish42                | Jun 2023 | Cleaned posts from the Blue Moon roleplaying forum.                                                                                               |
| [PIPPA](https://huggingface.co/datasets/kingbri/PIPPA-shareGPT)                                          | 16.8k    | Gosling et al., kingbri | Aug 2023 | Deduplicated version of Pygmalion's [PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA) in ShareGPT format.                                |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara)                                               | 16k      | LDJnr                   | Dec 2023 | Focuses on diverse information across multiple domains with multi-turn conversations.                                                             |
| [RPGPT_PublicDomain-alpaca](https://huggingface.co/datasets/practical-dreamer/RPGPT_PublicDomain-alpaca) | 4.26k    | Practical Dreamer       | May 2023 | Synthetic dialogues of public domain characters in roleplay format using [build-a-dataset](https://github.com/practical-dreamer/build-a-dataset). |
| [Pure-Dove](https://huggingface.co/datasets/LDJnr/Pure-Dove)                                             | 3.86k    | LDJnr                   | Sep 2023 | Highly filtered multi-turn conversations between GPT-4 and real humans.                                                                           |
| [Opus Samantha](https://huggingface.co/datasets/macadeliccc/opus_samantha)                               | 1.85k    | macadelicc              | Apr 2024 | Multi-turn conversations with Claude 3 Opus.                                                                                                      |
| [LimaRP-augmented](https://huggingface.co/datasets/grimulkan/LimaRP-augmented)                           | 804      | lemonilia, grimulkan    | Jan 2024 | Enhanced version of LimaRP with human roleplaying conversations.                                                                                  |

### 🤖 Agent & Function Calling

Function calling allows LLMs to execute predefined functions based on user prompts, enabling integration with external systems and performing complex tasks.

| **Dataset**                                                                                           | **Size** | **Authors**     | **Date**    | **Description**                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------- | -------- | --------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)     | 113k     | Sahil Chaudhary | Sep 2023    | Instruction-answer pairs in multiple languages. [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml) variant available without conversation tags. |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)     | 60k      | Salesforce      | Jun 2024    | Created using a pipeline designed for verifiable function-calling data.                                                                                                                               |
| [Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN)                                     | 34.4k    | internlm        | Mar 2024    | Combines AgentInstruct, ToolBench, and ShareGPT datasets for training in tool use and function calling.                                                                                               |
| [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 11.5k    | NousResearch    | August 2024 | This dataset is the compilation of structured output and function calling data used in the Hermes 2 Pro series of models.                                                                             |

---

## ⚖️ Preference Alignment Datasets

Preference datasets for Direct Preference Optimization (DPO) are essential for aligning AI systems with human values and expectations. They improve performance, reduce biases, and enable personalization and effective evaluation.

| **Dataset**                                                                                                                        | **Size** | **Authors**      | **Date**   | **Description**                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ultrafeedback_binarized_cleaned](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned)                         | 186k     | Allenai          | Nov 2023   | One of the bits of magic behind the Zephyr model.                                                                                                                                                                                                                                                                                                                     |
| [ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 61k      | Bartolome et al. | March 2024 | This dataset is the recommended and preferred dataset by Argilla to use when fine-tuning on UltraFeedback.                                                                                                                                                                                                                                                            |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)                                                                      | 37k      | Dong et al.      | Nov 2023   | HelpSteer is an open-source Helpfulness Dataset (CC-BY-4.0) that supports aligning models to become more helpful, factually correct and coherent, while being adjustable in terms of the complexity and verbosity of its responses.                                                                                                                                   |
| [Capybara-Preferences](https://huggingface.co/datasets/argilla/Capybara-Preferences)                                               | 15k      | Argilla          | April 2024 | This dataset builds on LDJnr/Capybara by creating a preference dataset from an instruction-following dataset, splitting the final assistant turn for alternative model responses, which are then critiqued by GPT-4 using UltraFeedback.                                                                                                                              |
| [distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)                         | 12k      | Argilla          | Jan 2024   | The dataset is a "distilabeled" version of the widely used dataset: [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs).                                                                                                                                                                                                                     |
| [Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K)                                                      | 10k      | Xin et al.       | June 2024  | Step-DPO is a method for improving the mathematical reasoning of large language models (LLMs).                                                                                                                                                                                                                                                                        |
| [py-dpo-v0.1](https://huggingface.co/datasets/jondurbin/py-dpo-v0.1)                                                               | 9k       | Jon Durbin       | Jan 2024   | This DPO dataset is designed to improve Python coding skills by using tested responses from the [Vezora/Tested-22k-Python-Alpaca](https://huggingface.co/datasets/Vezora/Tested-22k-Python-Alpaca) dataset as "chosen" values, while "rejected" values, generated from airoboros-l2-13b-3.1 and bagel-7b-v0.1, are considered lower quality, with duplicates removed. |
| [prm_dpo_pairs_cleaned](https://huggingface.co/datasets/M4-ai/prm_dpo_pairs_cleaned)                                               | 8k       | M4-ai            | April 2024 | The dataset was created by cleaning and deduplicating M4-ai/prm_dpo_pairs, removing incorrect completions and about 3,000 duplicate examples, resulting in a high-quality dataset for training a robust math language model.                                                                                                                                          |
| [distilabel-capybara-dpo-7k-binarized](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized)               | 7k       | Argilla          | March 2024 | DPO dataset built with distilabel atop the awesome LDJnr/Capybara.                                                                                                                                                                                                                                                                                                    |
| [distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo)                           | 2k       | Argilla          | Nov 2023   | Math related DPO dataset by Argilla                                                                                                                                                                                                                                                                                                                                   |
| [contextual-dpo-v0.1](https://huggingface.co/datasets/jondurbin/contextual-dpo-v0.1)                                               | 1,3k     | Jon Durbin       | Jan 2024   | This is a dataset meant to enhance adherence to provided context (e.g., for RAG applications) and reduce hallucinations, specifically using the airoboros context-obedient question answer format                                                                                                                                                                     |
| [gutenberg-dpo-v0.1](https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1)                                                 | 1k       | Jon Durbin       | Jan 2024   | This is a dataset meant to enhance novel writing capabilities of LLMs, by using public domain books from [Project Gutenberg](https://gutenberg.org/)                                                                                                                                                                                                                  |
| [truthy-dpo-v0.1](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1)                                                       | 1k       | Jon Durbin       | June 2024  | Truthy DPO is a dataset aimed at improving the truthfulness of LLMs while maintaining immersive roleplay by focusing on corporeal, spatial, temporal awareness, and correcting common misconceptions.                                                                                                                                                                 |
| [toxic-dpo-v0.2](https://huggingface.co/datasets/unalignment/toxic-dpo-v0.2)                                                       | 541      | Unalignment      | Jan 2024   | The Toxic-DPO dataset contains harmful and toxic content intended to demonstrate how direct-preference-optimization (DPO) can de-censor a model, with usage restricted to lawful, non-malicious academic or research purposes, and users assuming full responsibility for its use.                                                                                    |

---

## 🛠️ Tools for Creating High-Quality Datasets

Building a valuable dataset is more about quality than quantity. Here are some tools and methods to help you curate effective datasets:

### 🧹 Data Deduplication and Cleaning

- **Exact Deduplication**: Remove identical entries by normalizing data (e.g., converting text to lowercase), generating hashes (like MD5 or SHA-256), and eliminating duplicates.
- **Fuzzy Deduplication**:
  - **MinHash**: Uses hashing, sorting, and Jaccard similarity for finding similar entries.
  - **BLOOM Filters**: Employs hashing and fixed-size vectors for approximate duplicate detection.
- **Decontamination**: Filter out samples that are too similar to test sets using exact or fuzzy methods.

### ✅ Evaluating Data Quality

- **Rule-Based Filtering**: Remove unwanted content using specific criteria, such as eliminating phrases like "As an AI assistant."
- [**Argilla**](https://argilla.io/): An open-source platform for collaborative data filtering and annotation.
- [**LLM-as-a-Judge**](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/llm_judge.ipynb): A Colab notebook to rate data quality using models like Mixtral-7x8B.
- [**Data Prep Kit**](https://github.com/IBM/data-prep-kit): A framework for preparing data for both code and language tasks, scalable from laptops to data centers.
- [**DataTrove**](https://github.com/huggingface/datatrove/): A Hugging Face library for large-scale data processing, used in creating [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

### 🛠️ Generating Additional Data

#### Supervised Fine-Tuning (SFT) Datasets

- [**Distilabel**](https://github.com/argilla-io/distilabel): Generates and augments data for SFT and DPO using techniques like UltraFeedback and DEITA.
- [**Auto Data**](https://github.com/Itachi-Uchiha581/Auto-Data): Automatically creates fine-tuning datasets using API models.
- [**Bonito**](https://github.com/BatsResearch/bonito): Generates synthetic instruction tuning datasets without GPT. Check out [AutoBonito](https://colab.research.google.com/drive/1l9zh_VX0X4ylbzpGckCjH5yEflFsLW04?usp=sharing) as well.
- [**Augmentoolkit**](https://github.com/e-p-armstrong/augmentoolkit): Converts raw text into datasets using various models.
- [**Magpie**](https://github.com/magpie-align/magpie): Efficient pipeline for generating high-quality synthetic data by prompting aligned LLMs.
- [**Genstruct**](https://huggingface.co/NousResearch/Genstruct-7B): An instruction generation model that creates valid instructions from raw data.
- [**DataDreamer**](https://datadreamer.dev/docs/latest/): A Python library for prompting and generating synthetic data.

#### Pre-Training Datasets

- [**llm-swarm**](https://github.com/huggingface/llm-swarm): Generates synthetic datasets for pretraining or fine-tuning using local LLMs or Hugging Face Inference Endpoints.
- [**Cosmopedia**](https://github.com/huggingface/cosmopedia): Code for creating the [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) dataset.
- [**textbook_quality**](https://github.com/VikParuchuri/textbook_quality): Generates textbook-quality data, inspired by Microsoft's Phi models.

### 🔍 Exploring and Visualizing Data

- [**sentence-transformers**](https://sbert.net/): A Python library for working with language embedding models.
- [**Lilac**](https://github.com/lilacai/lilac): Curates better data for LLMs, used by organizations like NousResearch, Databricks, Cohere, and Alignment Lab AI.
- [**Nomic Atlas**](https://github.com/nomic-ai/nomic): Interact with and gain insights from instructed data while storing embeddings.
- [**text-clustering**](https://github.com/huggingface/text-clustering): A Hugging Face framework for grouping similar textual data.
- [**BunkaTopics**](https://github.com/charlesdedampierre/BunkaTopics): Tools for data cleaning and visualizing topic models.
- [**Autolabel**](https://github.com/refuel-ai/autolabel): Automatically labels data using popular language models.

### 🌐 Data Scraping

- [**Trafilatura**](https://github.com/adbar/trafilatura): A Python and command-line tool for extracting text and metadata from the web, used to create [RefinedWeb](https://arxiv.org/abs/2306.01116).
- [**Marker**](https://github.com/VikParuchuri/marker): Quickly converts PDFs into markdown text.

---

## Conclusion

Building effective LLMs requires high-quality data at every stage, from pre-training to fine-tuning and preference alignment. By leveraging the datasets and tools mentioned in this guide, you can enhance your models’ capabilities and ensure they perform well across a variety of tasks.

Feel free to explore these resources and integrate them into your workflow to create more robust and capable language models. Happy modeling!
