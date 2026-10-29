# Recommendation

**Selected variant: `prompt_a`** (ready to ship).

In plain words: `prompt_a` won 4 of 4 questions with 0 vetoed answer(s). `prompt_b` had 4 vetoed answer(s) and 4 banned claim(s).

LLM review: replay, model `gpt-4.1-2025-04-14`, prompt `comparative_review` v3 (sha256 53c7c63aa824).

## Summary

| variant | query wins | vetoes | high-risk vetoes | policy_violation | unsupported_claim | overconfident_tone | missed_key_fact | irrelevant_answer | retrieval_miss | mean grounding | mean faithfulness | mean clarity | high-risk LLM warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `prompt_a` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.876 | 5.0 | 5.0 | 0 |
| `prompt_b` | 0 | 4 | 2 | 4 | 2 | 1 | 3 | 0 | 0 | 0.502 | 1.0 | 2.25 | 2 |

## Top reasons

1. `prompt_a`: 0 high-risk veto(es), 0 veto(es) in total, 0 high-risk LLM warning(s), 4 query win(s), mean tie-break score 0.9628.
2. `prompt_b`: 2 high-risk veto(es), 4 veto(es) in total, 2 high-risk LLM warning(s), 0 query win(s), mean tie-break score 0.0068.

Ranking order: fewest high-risk vetoes, then fewest vetoes, then most wins, then mean score, then name. LLM warnings are ship blockers on the selected variant, never vetoes.

## Per-query reasoning

### Q1 (medium risk): pick `prompt_a`, only safe candidate

- `prompt_a`: clean; score 0.9436
- `prompt_b`: vetoed by policy_violation; also overconfident_tone; score 0.3501
- LLM winner: `prompt_a`. LLM note (unverified): "`prompt_a` accurately reflects the evidence from D2, stating that most withdrawal reviews complete within 24 hours but may take longer if more verification is needed. `prompt_b` overclaims by guaranteeing clearance within 24 hours and stating reviews are usually instant, which is not supported by the evidence."

### Q2 (high risk): pick `prompt_a`, only safe candidate

- `prompt_a`: clean; score 0.9307
- `prompt_b`: vetoed by policy_violation; also missed_key_fact; score 0.0125; LLM warning: faithfulness 1 <= 2, LLM overclaim flag
- LLM winner: `prompt_a`. LLM note (unverified): "`prompt_a` is fully supported by D1, which states that support agents cannot manually view existing passwords and directs users to the password reset process. `prompt_b` is incorrect and contradicts the evidence by claiming support can send the current password."

### Q3 (high risk): pick `prompt_a`, only safe candidate

- `prompt_a`: clean; score 0.9769
- `prompt_b`: vetoed by policy_violation, unsupported_claim; also missed_key_fact; score -0.1892; LLM warning: faithfulness 1 <= 2, LLM overclaim flag
- LLM winner: `prompt_a`. LLM note (unverified): "`prompt_a` is fully faithful to D3, which explicitly states that screenshots are not accepted and specifies acceptable documents. `prompt_b` is incorrect and unsupported, as it claims screenshots are acceptable if details are visible, which contradicts the evidence."

### Q4 (medium risk): pick `prompt_a`, only safe candidate

- `prompt_a`: clean; score 1.0
- `prompt_b`: vetoed by policy_violation; also missed_key_fact, unsupported_claim; score -0.1463
- LLM winner: `prompt_a`. LLM note (unverified): "`prompt_a` is directly supported by D4, which states demo accounts do not support real deposits or withdrawals. `prompt_b` invents the possibility of withdrawal after upgrading and verification, which is not mentioned in the evidence."

## Clarity versus safety

No query where a clearer answer lost on safety.

## Known limitations

- Banned-claim detection is lexical. It handles word order and simple negation, but misses synonyms ("view" for "see"), claims split across sentences, and implied claims ("the money is on its way" implying approval).
- Grounding is token overlap with the retrieved evidence. A sentence that recombines evidence words into a false statement can still score well; only numbers are checked exactly.
- Retrieval is BM25 over a handful of documents. A paraphrased question with no shared words can retrieve the wrong document; retrieval_miss marks this and is not charged to the answer.
- The LLM review is one call with one model and no human calibration set, so its scores are a second opinion, not ground truth. Blind labels and shuffled order reduce name and position bias; they do not remove model bias.
- Thresholds and weights in config/eval.toml were set by reasoning about these fixtures, not fitted on labelled data.
- The harness ranks variants. Shipping is a human decision; a "ready to ship" line only means no configured blocker fired.
