# Transaction Pattern Simulator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/55qp5gjzz4/badge.svg)](https://ask.thili.ai/b/55qp5gjzz4)

This project, the Transaction Pattern Simulator, helps financial analysts by simulating sequences of customer transactions. It's designed to test fraud detection systems or model customer behavior patterns within the Banking & Finance domain. By applying frequent itemset mining and language model fine-tuning, you'll generate realistic transaction streams.

## What you are building

**Goal.** You will build a Transaction Pattern Simulator that simulates sequences of customer transactions to test fraud detection systems or model customer behavior patterns.

**Done when.** The project is done when you can use frequent itemset mining on transaction sequences, apply language model continued pretraining for generation, and successfully generate plausible transaction sequences.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Represent anonymized financial transactions as ordered event sequences.**  
      Hint: Structure your transaction data into sequences, preparing it for `lfretail-lfmarketbasket`.
- [ ] **2. Discover frequent transaction patterns using `thilimarketbasket`.**  
      Hint: Apply the `lfretail-lfmarketbasket` technique to identify common transaction patterns.
- [ ] **3. Fine-tune an LM on these sequences to generate new, plausible transaction streams.**  
      Hint: Use the language model continued pretraining technique on your transaction sequences.
- [ ] **4. Evaluate the generated sequences for similarity to real patterns.**  
      Hint: Compare the frequent patterns discovered in real data with those in generated sequences.
- [ ] **5. Implement a CLI to generate sequences based on a starting set of transactions.**  
      Hint: Build a command-line interface that uses your fine-tuned LM to generate sequences.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the evaluation to provide quantitative metrics of similarity between real and generated patterns.
- Extend the CLI to allow users to specify constraints or desired characteristics for the generated sequences (e.g., minimum length, presence of specific transaction types).
- Implement a feature to save generated sequences to a file for further analysis.
