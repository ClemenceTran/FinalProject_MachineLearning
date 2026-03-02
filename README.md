# FinalProject_MachineLearning
# Summary
    Using DistilGPT2, QuestCrafter is a controllable text generating system that has been refined to produce fantasy quests based on structured control tokens
    <LEVEL> <CONFIGURING> <TONE> <LENGTH>
    The project compares a baseline pretrained model against a fine-tuned model using both automatic and human evaluation.

# How to run ? 
    ## Install Dependencies: pip install -r requirements.txt
    ## Check evaluation for baseline model: python evaluation_baseline.py        or      python3 evaluation_baseline.py
    ## Check evaluation for finetuned model: python evaluation_finetuned.py        or      python3 evaluation_finetuned.py
    ## Run the Demo Application: python -m streamlit run demoapp.py         or          streamlit run demoapp.py      

    * The demo contains: Select model (Baseline or Fine-Tuned), Choose Level, Setting, Tone, Length, Provide additional quest instructions, Generate a fantasy quest.

# Results Summary
    ## Automatic Evaluation Results
    | Model                 | Validation Loss | Perplexity | Distinct-1 | Distinct-2 |
    |-----------------------|-----------------|------------|------------|------------|
    | Baseline (DistilGPT2) | 3.1764          | 23.96      | 0.2708     | 0.7762     |
    | Fine-Tuned            | 1.4080          | 4.09       | 0.1528     | 0.5140     |
    
    => The fine-tuned model greatly reduces perplexity compared to the baseline, showing improved coherence and dataset alignment. Although diversity slightly decreases, the generated quests are more structured and consistent.

    ## Human Evaluation Results
    | Model                 | Coherence (1–5) | Faithfulness (1–5) | Creativity (1–5) |
    |-----------------------|-----------------|--------------------|------------------|
    | Baseline (DistilGPT2) | 1.0             | 1.0                | 1.0              |
    | Fine-Tuned            | 3.0             | 2.0                | 2.0              |

    => The fine-tuned model generates more coherent and readable stories than the baseline. However, it only partially follows the specified control tokens, leading to moderate faithfulness scores.




