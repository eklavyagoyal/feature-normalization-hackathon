# Portfolio & Interview Guide

> Ready-to-use material for CVs, resumes, LinkedIn, project descriptions, and technical interviews.

---

## Resume Bullets

Pick 2-3 of these for a resume or CV:

1. **Engineered a hybrid extraction pipeline** processing 7.86M product rows in ~5 min at $0 API cost, achieving 78.63% exact match accuracy against a constrained taxonomy of 16K+ allowed values — winning first place (€500) in a competitive hackathon

2. **Designed an 8-layer deterministic waterfall** (Aho-Corasick trie matching, structured description parsing, dimension disambiguation, regex numeric extraction) that handles ~85% of predictions before lightweight ML fallback

3. **Built a strict dual-classifier ensemble** (word + char n-gram TF-IDF/SGD) requiring agreement before overriding deterministic predictions — prioritizing precision over recall to prevent noisy ML overrides

4. **Implemented taxonomy-constrained output validation** with multi-stage snapping (exact → case-insensitive → fuzzy), ensuring zero invalid predictions across 7.86M output rows

5. **Demonstrated 3,800× cost advantage** over LLM-based approaches at 200M product scale ($0.27 compute vs $1,000+ API), with fully deterministic reproducibility

---

## LinkedIn / Project Description Variants

### Variant 1 (Technical)

> Built a high-throughput feature normalization engine that extracts structured product attributes from unstructured German catalog text, constrained to a predefined taxonomy. The system uses a multi-layer deterministic waterfall (Aho-Corasick trie matching, structured parsing, error-analysis-driven domain rules) with a lightweight ML ensemble fallback — achieving 78.63% exact match accuracy on 7.86M rows at $0 API cost. Won first place (€500) at a hackathon by demonstrating that pattern-based methods with targeted rules outperform LLM-based approaches when cost and throughput are evaluation criteria.

### Variant 2 (Product-Oriented)

> Won first place in a competitive hackathon by building a zero-cost product attribute extraction pipeline that normalizes unstructured German text into structured taxonomy features. The system processes 3.1M products (~26,600 rows/sec) using a hybrid approach: deterministic extraction handles ~85% of predictions at zero cost, with an ML ensemble covering the remainder. At 200M product scale, the solution costs $0.27 in compute — versus $1,000+ for LLM-based approaches — while maintaining 78.63% accuracy.

---

## Interview Answers

### "Tell me about this project" (2 min)

> This was a hackathon project where we had 2.5 days to build a system that normalizes unstructured product catalog text into structured features. The input is German product titles and descriptions — things like "Sechskantschraube DIN 933 M12x80 Edelstahl A2" — and the output is structured attributes like Material = "Edelstahl (A2)", Thread size = "M 12", Length = "80 mm", each constrained to a predefined taxonomy of allowed values.
>
> The interesting constraint was the evaluation rubric: 40% accuracy, 40% cost, 20% throughput. Most teams reached for LLMs, which gives decent accuracy but costs a lot at scale. I took the opposite approach — build a deterministic extraction pipeline first, using pattern matching, Aho-Corasick tries, structured description parsing, and domain-specific rules. The ML component is a lightweight TF-IDF ensemble that only kicks in for rows where the deterministic layers have low confidence.
>
> The result was 78.63% exact match accuracy on 7.86M rows, $0 API cost, ~5 minutes runtime. It won first place because it dominated on the cost and throughput criteria while staying competitive on accuracy. The key engineering judgment was recognizing that for structured extraction from semi-structured text, targeted rules beat general-purpose language models — especially when you iterate on error analysis to find the highest-impact patterns.

### "Walk me through the technical architecture" (3-5 min)

> The system is a multi-layer waterfall pipeline. Each layer attempts to extract a feature value; only unresolved rows cascade to the next layer.
>
> The first layer is a train title lookup — if we've seen the exact same (feature_name, title) pair in training data, we use the most common value. This alone resolves about 14% of rows at very high precision.
>
> Then we have domain-specific rules — these were built iteratively by analyzing validation errors. For example, we discovered that for the "Material" feature, the system was predicting "Edelstahl" when the correct answer was "Edelstahl (A2)" because both "Edelstahl" and "A2" appeared in the text but as separate tokens. So we added compound matching logic that checks if both the base material and the grade code are present.
>
> The structured description parser handles products that have "Technische Daten" sections with explicit key:value pairs. This is high-precision extraction — we just need to match the key to our feature name, with some alias handling for German variants.
>
> For categorical features, we compile all ~16K unique taxonomy values into an Aho-Corasick trie. This gives us O(text length) multi-pattern matching — one pass through the text finds all matching values. We then rank by specificity: prefer "Edelstahl (A2)" over "Edelstahl" because it has more tokens matching.
>
> For numeric features, we auto-generate regex patterns from the taxonomy's units. If the taxonomy says a feature uses "mm" as its unit, we search for `(\d+)\s*mm` patterns. We prioritize title matches over description matches because titles are higher signal density, and we use positional heuristics for screw features where the first number is typically the diameter and the second is the length.
>
> After all deterministic layers, unresolved categorical rows go to a TF-IDF + SGD ensemble. We have two independently trained classifiers — one using word n-grams, one using character n-grams. Both must agree before overriding a deterministic prediction. This strict agreement requirement prevents noisy ML overrides.
>
> Finally, every prediction passes through a taxonomy validator that snaps values to the allowed set via exact match, case-insensitive match, or fuzzy matching. This ensures zero invalid outputs.
>
> The whole pipeline runs on CPU with standard Python libraries — pandas, scikit-learn, pyahocorasick. No GPU, no API calls, ~5 minutes for 7.86M rows.

### "Why did this win the hackathon?" (1-2 min)

> The rubric was 40% accuracy, 40% cost, and 20% throughput. Most teams optimized for accuracy using LLMs, which meant high API costs at scale. My strategy was to recognize that satisfying the cost and throughput criteria — which together are 60% of the rubric — was as important as accuracy.
>
> A deterministic-first approach naturally dominates both: zero API cost and maximum throughput. The engineering challenge was making the deterministic pipeline accurate enough to stay competitive on the accuracy criterion.
>
> The key technique was iterative error analysis. After each pipeline iteration, I'd look at the worst-performing features on validation, diagnose the root cause, and build a targeted fix. For example, compound material matching added +2% accuracy in a single rule. Dimension disambiguation (HxBxT positional mapping) fixed an entire class of errors. Each domain rule was justified by measured lift.
>
> The result was a system that was slightly behind on pure accuracy (some LLM approaches might hit 80-82%) but so far ahead on cost and throughput that the overall weighted score was highest. At 200M product scale, the cost difference is $0.27 versus over $1,000 — that's a 3,800× advantage.

---

## Project Positioning

For different audiences, emphasize different aspects:

| Audience | Emphasize |
|:---|:---|
| **ML/AI roles** | Multi-layer extraction, ensemble design, error-analysis-driven iteration, taxonomy-constrained output |
| **Data engineering** | Throughput (26K rows/sec), Aho-Corasick O(n) matching, scaling to 200M products, deterministic reproducibility |
| **Product/applied ML** | Cost-aware design, production-readiness, business impact of $0 vs $1K+, practical engineering over hype |
| **Startup/generalist** | Hackathon winner, built in 2.5 days, pragmatic approach, strong engineering judgment under constraints |
| **Search/information extraction** | Taxonomy-constrained extraction, German NLP, structured parsing, multi-pattern matching |

---

## Key Talking Points

**On engineering judgment:**
> "I chose deterministic extraction not because I can't use LLMs, but because I understood the evaluation rubric. Cost was 40% of the score. The engineering judgment was recognizing when not to use the most powerful tool."

**On iterative improvement:**
> "Each domain rule was discovered through error analysis, not guessed upfront. I'd run the pipeline on validation, sort features by accuracy, diagnose the worst ones, and build targeted fixes. This is how production ML systems actually improve."

**On the hybrid approach:**
> "The ML ensemble is deliberately conservative — both classifiers must agree before overriding a deterministic answer. This is a precision-over-recall tradeoff. In production, wrong overrides erode trust faster than missing predictions."

**On scaling:**
> "At 200M products, this pipeline runs for ~33 minutes and costs $0.27 in compute. An LLM approach would cost over $1,000 in API calls alone. That's the difference between 'runs in production' and 'runs as a demo'."

**On the hackathon:**
> "I built this in 2.5 days. The code could be cleaner, but every engineering decision was deliberate. I didn't reach for the most complex solution — I reached for the most appropriate one."
