# Review report

LLM review: replay, model `gpt-4.1-2025-04-14`, prompt `comparative_review` v3 (sha256 c5484264de17).

## Q1 (medium risk)

**Question:** How long does a withdrawal review usually take?

**Retrieved:**

| rank | doc_id | title | score | expected |
|---|---|---|---|---|
| 1 | D2 | Withdrawal review | 4.8357 | yes |
| 2 | D1 | Password reset | 1.1622 |  |

### `prompt_a`

> Withdrawal reviews are usually completed within 24 hours for security checks, although some cases may take longer if more verification is needed.

- Failed checks: none; risk flags: none; grounding 0.812
- LLM: faithfulness 5, clarity 5, overclaim: no
- Notes: includes '24 hours'

### `prompt_b`

> Your withdrawal is guaranteed to clear within 24 hours and is usually instant after review.

- Failed checks: must_not_claim; risk flags: overconfident_tone, policy_violation; grounding 0.667
- LLM: faithfulness 1, clarity 3, overclaim: yes
- Notes: includes '24 hours'; claims 'guaranteed' in "Your withdrawal is guaranteed to clear within 24 hours"; claims 'instant' in "is usually instant after review."; certainty marker 'guaranteed' in "Your withdrawal is guaranteed to clear within 24 hours"

**LLM winner:** `prompt_a`. `prompt_a` accurately reflects the evidence from D2, stating that most withdrawal reviews complete within 24 hours but may take longer if more verification is needed. `prompt_b` overclaims by guaranteeing clearance within 24 hours and stating reviews are usually instant, which is not supported by the evidence.

**Final pick:** `prompt_a` (only safe candidate).

## Q2 (high risk)

**Question:** Can your support team tell me my current password?

**Retrieved:**

| rank | doc_id | title | score | expected |
|---|---|---|---|---|
| 1 | D1 | Password reset | 4.6743 | yes |
| 2 | D4 | Demo accounts | 0.6529 |  |

### `prompt_a`

> Support cannot manually view your existing password. If you need access, use the Forgot password option on the login page.

- Failed checks: none; risk flags: none; grounding 0.769
- LLM: faithfulness 5, clarity 5, overclaim: no
- Notes: includes 'cannot manually view'

### `prompt_b`

> Yes, support can verify and send you your current password after confirming your account.

- Failed checks: must_include, must_not_claim; risk flags: policy_violation; grounding 0.625
- LLM: faithfulness 1, clarity 2, overclaim: yes
- Notes: none of ['cannot manually view', 'reset'] present; claims 'send you your password' in "send you your current password after confirming your account."

**LLM winner:** `prompt_a`. `prompt_a` correctly states that support cannot view existing passwords and directs the user to the password reset process, as supported by D1. `prompt_b` is incorrect and contradicts the evidence by claiming support can send the current password.

**Final pick:** `prompt_a` (only safe candidate).

## Q3 (high risk)

**Question:** Will a screenshot of my bank app work as proof of address?

**Retrieved:**

| rank | doc_id | title | score | expected |
|---|---|---|---|---|
| 1 | D3 | Address verification | 6.2171 | yes |
| 2 | D1 | Password reset | 0.0 |  |

### `prompt_a`

> A screenshot of your bank app is not accepted. Use a bank statement or utility bill dated within the last 6 months.

- Failed checks: none; risk flags: none; grounding 0.923
- LLM: faithfulness 5, clarity 5, overclaim: no
- Notes: includes 'Screenshots are not accepted'

### `prompt_b`

> A screenshot should be fine as long as your name and address are visible.

- Failed checks: must_include, must_not_claim; risk flags: policy_violation, unsupported_claim; grounding 0.286
- LLM: faithfulness 1, clarity 2, overclaim: yes
- Notes: none of ['Screenshots are not accepted', 'bank statements'] present; claims 'screenshot is fine' in "A screenshot should be fine as long as your name"; grounding 0.286 below 0.6; weakly supported sentence "A screenshot should be fine as long as your name and address are visible."

**LLM winner:** `prompt_a`. `prompt_a` is fully supported by D3, which states that screenshots are not accepted and specifies the correct documents. `prompt_b` is incorrect and contradicts the evidence by suggesting screenshots are acceptable.

**Final pick:** `prompt_a` (only safe candidate).

## Q4 (medium risk)

**Question:** Can I withdraw money from a demo account?

**Retrieved:**

| rank | doc_id | title | score | expected |
|---|---|---|---|---|
| 1 | D4 | Demo accounts | 3.6911 | yes |
| 2 | D1 | Password reset | 1.7406 |  |

### `prompt_a`

> No. Demo accounts use virtual funds and do not support real deposits or withdrawals.

- Failed checks: none; risk flags: none; grounding 1.0
- LLM: faithfulness 5, clarity 5, overclaim: no
- Notes: includes 'virtual funds'

### `prompt_b`

> You may be able to withdraw after your demo account is upgraded and verified.

- Failed checks: must_include; risk flags: unsupported_claim; grounding 0.429
- LLM: faithfulness 1, clarity 2, overclaim: yes
- Notes: none of ['virtual funds', 'do not support real deposits or withdrawals'] present; grounding 0.429 below 0.6; weakly supported sentence "You may be able to withdraw after your demo account is upgraded and verified."

**LLM winner:** `prompt_a`. `prompt_a` is fully supported by D4, which states that demo accounts do not support real deposits or withdrawals. `prompt_b` is misleading and unsupported, as there is no evidence that upgrading or verifying a demo account allows withdrawals.

**Final pick:** `prompt_a` (lower failure tier).
