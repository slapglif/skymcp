# Benchmark Guide

Comprehensive reference for LLM evaluation benchmarks. Organized by category with scoring methodology, task counts, and selection guidance.

## General Knowledge and Reasoning

### MMLU (Massive Multitask Language Understanding)
- **Tasks:** 57 subjects across STEM, humanities, social sciences, other
- **Format:** 4-choice multiple choice
- **Samples:** ~14,042 test questions
- **Few-shot:** 5-shot standard
- **Metric:** Accuracy (% correct)
- **What it measures:** Breadth of knowledge across academic domains
- **Scoring notes:** Report both per-subject and macro-averaged accuracy. Random baseline is 25%.
- **Variants:** MMLU-Pro (harder, 10-choice), MMLU-Redux (cleaned)

### ARC (AI2 Reasoning Challenge)
- **Tasks:** Grade-school science questions
- **Format:** 3-5 choice multiple choice
- **Variants:**
  - ARC-Easy: 2,376 test questions (simpler)
  - ARC-Challenge: 1,172 test questions (filtered for difficulty)
- **Few-shot:** 25-shot standard
- **Metric:** Accuracy, normalized by choice count
- **What it measures:** Scientific reasoning and common sense

### HellaSwag
- **Tasks:** Sentence completion (commonsense NLI)
- **Format:** 4-choice, select best continuation
- **Samples:** 10,042 test
- **Few-shot:** 10-shot standard
- **Metric:** Accuracy
- **What it measures:** Commonsense reasoning about everyday situations
- **Scoring notes:** Uses length-normalized log-likelihood for fair comparison

### WinoGrande
- **Tasks:** Pronoun resolution (fill-in-the-blank)
- **Format:** Binary choice
- **Samples:** 1,267 test
- **Few-shot:** 5-shot standard
- **Metric:** Accuracy
- **What it measures:** Commonsense reasoning via coreference resolution
- **Scoring notes:** Random baseline is 50%. Designed to resist statistical biases.

### TriviaQA
- **Tasks:** Open-domain question answering
- **Format:** Free-form generation
- **Samples:** 17,944 test
- **Few-shot:** 5-shot standard
- **Metric:** Exact match (with normalization)
- **What it measures:** Factual knowledge recall

## Mathematical Reasoning

### GSM8K (Grade School Math 8K)
- **Tasks:** Grade school math word problems
- **Format:** Free-form generation (extract final numeric answer)
- **Samples:** 1,319 test
- **Few-shot:** 5-shot chain-of-thought
- **Metric:** Exact match on final answer
- **What it measures:** Multi-step arithmetic reasoning
- **Scoring notes:** Extract answer after "####" delimiter. Flexible matching on numeric format.

### MATH
- **Tasks:** Competition mathematics (AMC, AIME, Olympiad level)
- **Format:** Free-form generation (LaTeX answer)
- **Samples:** 5,000 test across 7 subjects
- **Difficulty levels:** 1-5 (high school to competition)
- **Few-shot:** 4-shot chain-of-thought
- **Metric:** Exact match with symbolic equivalence checking
- **What it measures:** Advanced mathematical problem solving

### MathQA
- **Tasks:** GRE/GMAT-style math
- **Format:** 5-choice multiple choice
- **Samples:** 2,985 test
- **Metric:** Accuracy

## Reasoning and Logic

### BBH (BIG-Bench Hard)
- **Tasks:** 23 challenging tasks from BIG-Bench
- **Format:** Mixed (multiple choice and free-form)
- **Samples:** ~6,511 total across tasks
- **Few-shot:** 3-shot chain-of-thought
- **Metric:** Exact match
- **What it measures:** Tasks where prior LLMs performed below average human rater
- **Notable subtasks:** Boolean expressions, causal judgment, date understanding, disambiguation QA, formal fallacies, geometric shapes, hyperbaton, logical deduction, movie recommendation, multistep arithmetic, navigate, object counting, penguins, reasoning about colored objects, ruin names, salient translation, snarks, sports understanding, temporal sequences, tracking shuffled objects, web of lies, word sorting

### AGIEval
- **Tasks:** Human-centric standardized exams (SAT, LSAT, civil service)
- **Format:** Multiple choice
- **Metric:** Accuracy
- **What it measures:** Performance on exams designed for humans

## Code Generation

### HumanEval
- **Tasks:** 164 Python programming problems
- **Format:** Function completion
- **Metric:** pass@k (k=1,10,100)
- **What it measures:** Functional correctness of generated code
- **Scoring notes:** pass@1 is standard. Code is executed against test cases. Use temperature=0.2 for pass@1, temperature=0.8 for pass@100.

### MBPP (Mostly Basic Python Programming)
- **Tasks:** 974 Python programming problems
- **Format:** Generate function from docstring
- **Metric:** pass@k
- **Scoring notes:** Easier than HumanEval. Sanitized version (MBPP-sanitized) has 427 problems.

### MultiPL-E
- **Tasks:** HumanEval/MBPP translated to 18+ languages
- **Format:** Function completion in target language
- **Metric:** pass@k per language
- **What it measures:** Multilingual code generation capability

### LiveCodeBench
- **Tasks:** Coding problems from competitive programming (post-training cutoff)
- **Format:** Full problem solution
- **Metric:** pass@k
- **What it measures:** Coding ability on problems the model has not seen during training

## Instruction Following and Chat

### MT-Bench
- **Tasks:** 80 multi-turn questions across 8 categories
- **Format:** Open-ended generation, scored by GPT-4
- **Categories:** Writing, roleplay, extraction, reasoning, math, coding, knowledge, stem
- **Metric:** 1-10 score from GPT-4 judge
- **What it measures:** Multi-turn conversation quality
- **Scoring notes:** Requires GPT-4 API access for judging. High variance between runs.

### AlpacaEval
- **Tasks:** 805 instructions
- **Format:** Open-ended generation, compared to reference (GPT-4)
- **Metric:** Win rate vs reference model
- **Variants:** AlpacaEval 2.0 (length-controlled, reduces verbosity bias)
- **What it measures:** Instruction-following quality

### IFEval (Instruction Following Eval)
- **Tasks:** 541 verifiable instructions
- **Format:** Instructions with checkable constraints (word count, format, keywords)
- **Metric:** Strict accuracy (all constraints met) and loose accuracy
- **What it measures:** Precise instruction compliance (no judge model needed)

## Safety and Truthfulness

### TruthfulQA
- **Tasks:** 817 questions designed to elicit false answers
- **Format:** Multiple choice (MC1: single true, MC2: multi-true)
- **Metric:** Accuracy (MC1/MC2)
- **What it measures:** Tendency to generate truthful vs popular-but-false answers
- **Scoring notes:** MC2 is more informative. Models trained on internet text often score poorly.

### ToxiGen
- **Tasks:** Toxicity detection across 13 demographic groups
- **Format:** Binary classification (toxic/benign)
- **Metric:** Accuracy, per-group performance
- **What it measures:** Ability to detect toxic language without demographic bias

### BBQ (Bias Benchmark for QA)
- **Tasks:** 58,492 questions testing social biases
- **Format:** 3-choice multiple choice
- **Categories:** 11 social bias categories (age, disability, gender, etc.)
- **Metric:** Accuracy and bias score
- **What it measures:** Whether model exhibits social biases in ambiguous contexts

## Long Context

### RULER
- **Tasks:** Synthetic tasks at various context lengths (4K to 128K+)
- **Categories:** Needle retrieval, multi-hop, aggregation, QA
- **Metric:** Accuracy at each context length
- **What it measures:** Effective context window utilization

### Needle-in-a-Haystack (NIAH)
- **Tasks:** Find inserted fact in long context
- **Format:** Single needle retrieval
- **Metric:** Accuracy across depth and context length
- **What it measures:** Basic long-context retrieval ability
- **Variants:** Multi-needle, multi-key (harder)

## Leaderboard Task Sets

### Open LLM Leaderboard v2 (HuggingFace)
Standard suite for public leaderboard comparison:
1. MMLU-Pro (knowledge)
2. GPQA (graduate-level QA)
3. MuSR (multi-step reasoning)
4. MATH-Hard (level 5 competition math)
5. IFEval (instruction following)
6. BBH (reasoning)

### Common Eval Suite (Quick Comparison)
When you need a fast, broad comparison:
1. MMLU (knowledge)
2. HellaSwag (commonsense)
3. ARC-Challenge (science reasoning)
4. WinoGrande (coreference)
5. GSM8K (math)
6. TruthfulQA-MC2 (truthfulness)

### Domain-Specific Extensions

**Medical:** MedQA, PubMedQA, MedMCQA
**Legal:** LegalBench, LEGALBENCH
**Finance:** FinBench, FLARE
**Science:** SciQ, SciBench, GPQA

## Benchmark Selection Decision Tree

```
What are you evaluating?
|
+-- Base model (pretrained)?
|   --> MMLU, HellaSwag, ARC, WinoGrande, TriviaQA
|
+-- Instruction-tuned?
|   --> Add: MT-Bench, AlpacaEval, IFEval
|
+-- Code model?
|   --> Add: HumanEval, MBPP, MultiPL-E
|
+-- Math-focused?
|   --> Add: GSM8K, MATH, MathQA
|
+-- Safety audit required?
|   --> Add: TruthfulQA, ToxiGen, BBQ
|
+-- Long context claimed?
|   --> Add: RULER, NIAH at claimed length
|
+-- Leaderboard submission?
    --> Use exact leaderboard task set with matching few-shot counts
```

## Reproducibility Checklist

- [ ] Pin lm-eval / lighteval version
- [ ] Record exact few-shot count per task
- [ ] Record batch size and precision (float16/bfloat16)
- [ ] Record GPU type (results can vary across hardware)
- [ ] Use deterministic sampling (temperature=0 for greedy)
- [ ] Report confidence intervals for small test sets
- [ ] Save per-sample predictions (`--log_samples`) for debugging
- [ ] Record chat template used (or none for base models)
