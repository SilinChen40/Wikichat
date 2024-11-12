# DS 5690 - Wikichat Paper Presentation

**Silin Chen, Holly Hou**

## Motivation - The Rise of Conversational AI and the Factuality Challenge

* **Widespread Use of LLMs**: Large Language Models (LLMs) like GPT-4 have become essential tools for chatbots, enabling natural and engaging conversations.

* **The Problem with Hallucination**: Despite their conversational strengths, LLMs often produce factually incorrect information, or "hallucinate," particularly on lesser-known topics or recent events.

* **Demand for Reliable AI**: As more applications adopt LLMs for real-world use cases (healthcare, education, knowledge-sharing), there is an urgent need for chatbots that provide accurate, up-to-date, and reliable information.

<p align="center" width="100%">
<img src="images\llm_hall.png" alt="" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>

## Problem Statement - Addressing Hallucination in LLMs: Accuracy vs. Engagement 

* **Factual Inaccuracy**: LLMs generate responses that are not always grounded in verified information, leading to potential misinformation.

* **Challenges with Tail and Recent Topics**:

  * Tail Topics: Niche subjects are often less represented in pre-trained data, increasing the likelihood of hallucination.

  * Recent Knowledge: LLMs cannot access events that occurred after their training cut-off, making them unreliable for current information.

* **Balancing Accuracy and Conversationality**: Efforts to ensure factual accuracy often come at the cost of conversational quality, making responses less engaging or overly simplified.

## Approach

1. **Retrieve**: The user query initiates a search in Wikipedia to gather relevant, up-to-date information.

2. **Generate**: An LLM generates a preliminary response based on the retrieved information.

3. **Extract and Verify Claims**: Each claim from the response is fact-checked using evidence from Wikipedia to filter out unverified statements.

4. **Draft and Refine**: The system drafts an initial response with only verified claims, then refines it to improve clarity, naturalness, and engagement.

## Implementation

The authors developed two versions of WikiChat (G3.5 and G4) and a distilled, smaller version (WikiChat L) to enhance efficiency. They used **ColBERT v2** for fast and accurate retrieval from Wikipedia, supporting reliable fact-checking. Performance was tested through **simulated dialogues**, demonstrating WikiChat’s high factual accuracy, especially on rare and recent topics.   

## Architecture Overview

WikiChat’s architecture is built around a **seven-step pipeline** designed to ensure factual accuracy and conversational quality. This pipeline goes beyond simple retrieval and generation, incorporating steps for filtering, claim verification, and response refinement to prevent hallucinations. Central to this process is the **DSPy pipeline**, which provides a modular framework for efficient and flexible integration of retrieval, generation, and fact-checking components.

<p align="center" width="100%">
<img src="images\wikipipeline.png" alt="" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

## Key Questions

1. How does WikiChat ensure factual accuracy in its responses while maintaining conversational quality?

WikiChat uses a seven-step pipeline with claim verification and response refinement to balance accuracy and engagement.
  
2. What techniques does WikiChat use to optimize efficiency, and how do these impact its real-world applicability?

WikiChat employs DSPy for modular efficiency and model distillation to reduce latency, making it practical for real-time use.

### Understanding WikiChat’s Unique Approach

* **Retrieve-then-Generate Approach**: Unlike previous models that either rely entirely on retrieval (giving dry, fact-based responses) or generation (risking hallucinations), WikiChat combines both methods. It retrieves accurate information and then generates responses, ensuring they are engaging yet factual.

* **Fact-Checking Pipeline**: Traditional LLM chatbots lack a structured verification process, leading to hallucinations. WikiChat’s approach includes a claim extraction and verification process, where each claim generated by the LLM is cross-checked with Wikipedia data. This makes WikiChat unique in its rigorous filtering of potentially inaccurate information.

* **Refinement Stage for Conversational Quality**: In contrast to fact-checking systems that sacrifice engagement for accuracy, WikiChat has a final refinement stage. This step is designed to improve the response's conversational flow, making it natural and user-friendly.

### Additional Approach

* **Model distillation**: WikiChat employs this method to create a smaller, optimized version of the LLaMA model, retaining accuracy while reducing latency and operational costs for real-world usability. The evaluation process combines human and model-based assessments, focusing on factual accuracy and conversational quality to ensure responses are both reliable and engaging.
* **Evaluation process**: It combines human and model-based assessments, focusing on factual accuracy and conversational quality to ensure responses are both reliable and engaging.

## Key Findings
1. **Claim Density**: WikiChat G4 generated more claims per turn (3.6 claims) compared to GPT-4 (2.5 claims), providing richer responses.

2. **Source Reliance**: WikiChat balanced reliance between LLM output (30%) and Wikipedia-based information (70%), strengthening accuracy.

3. **Fact-Checking Rejections**: About one-third of claims, particularly in niche and recent topics, were filtered out, significantly reducing hallucinations.

4. **Refinement Improvements**: The final refinement stage boosted conversational quality and temporal accuracy, particularly for niche and recent topics.

5. **Trustworthy “I Don’t Know” Responses**: For unavailable information, WikiChat responds with “I don’t know,” enhancing user trust.  

## Critical Analysis: Key Limitations and Opportunities for Improvement

* **Reliance on Wikipedia as a Primary Knowledge Source**: While Wikipedia is a widely trusted and frequently updated source, its coverage may be insufficient for specialized or emerging topics, such as certain areas in medicine or law. This reliance could limit WikiChat’s effectiveness in domains where comprehensive, specialized knowledge is essential. Expanding WikiChat’s capacity to draw from other reliable databases could make it more versatile and adaptable.

* **Latency Challenges Due to the 7-Stage Pipeline**: WikiChat’s multi-step approach, involving claim extraction, verification, and refinement, could lead to slower response times, especially in high-demand, real-time scenarios. Although this design enhances factual accuracy, future work on optimizing or parallelizing the process could improve response speed without sacrificing reliability.

## Impact of WikiChat on the AI Landscape

* **Reducing Hallucination in LLMs**: By implementing the approach outlined in WikiChat, we have significantly decreased hallucination in our large language model (LLM) responses. Grounding responses in verifiable information from Wikipedia has enhanced the factual accuracy of the chatbot, resulting in more reliable and trustworthy interactions.

* **Efficiency Gains with DSPy Pipeline**: Leveraging the DSPy pipeline has streamlined the process, making the entire pipeline more efficient. This integration allows for faster retrieval, claim extraction, and verification, reducing latency without compromising on the accuracy of the generated responses.

* **Extending WikiChat’s Approach for Broader Applications:**: Inspired by WikiChat, we conducted our own experiments combining Retrieval-Augmented Generation (RAG) and fact-checking. This hybrid approach has demonstrated its potential for a variety of tasks beyond traditional Q&A, opening up possibilities for new applications where both accuracy and conversational quality are critical.

## References

1. Semnani, Shima Jazayeri, Victor Zhongkai Yao, Hongchang Zhang, and Monica S. Lam. "WikiChat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia." arXiv preprint arXiv:2305.14292 (2023).
2. Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, ZiweiJi, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity.
3. Jianzhu Bao, Rui Wang, Yasheng Wang, Aixin Sun, Yitong Li, Fei Mi, and Ruifeng Xu. 2023. A synthetic data generation framework for grounded dialogues. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10866–10882, Toronto, Canada. Association for Computational Linguistics.


## Resources

- [WikiChat GitHub Repository](https://github.com/stanford-oval/WikiChat): Access the code and implementation details for WikiChat.
- [WikiChat Paper on arXiv](https://arxiv.org/abs/2305.14292): Read the full research paper for an in-depth understanding of the approach and methodology.
















