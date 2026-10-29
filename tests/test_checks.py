from rag_eval.checks import check_answer, claim_made, grounding, Evidence, phrase_present
from rag_eval.config import ChecksConfig, load_config
from rag_eval.schema import Query
from rag_eval.text import normalize, stem

CFG = load_config(None).checks.model_copy(update={
    "certainty_markers": ["guarantee", "guaranteed", "definitely"],
    "refusal_markers": ["human agent", "i do not have"],
})
EVIDENCE = [{"doc_id": "D1", "title": "Password reset", "text": "Support staff cannot see your password. Reset links expire after 30 minutes."}]


def _query(**kw) -> Query:
    base = dict(query_id="X", user_question="Can support see my password?", expected_doc_ids=["D1"],
                must_include_any=["reset link"], must_not_claim=["support can see your password"], risk_level="high")
    return Query(**{**base, **kw})


def test_normalize_handles_curly_quotes_contractions_and_thousands():
    assert normalize("We can’t do $2,000") == "we can not do $2000"
    assert stem("approved") == stem("approve") == stem("approves")


def test_banned_claim_detected_in_plain_statement():
    assert claim_made("Yes, support can see your password.", "support can see your password")


def test_negated_claim_is_not_a_violation():
    assert claim_made("Support staff cannot see your password.", "support can see your password") is None
    assert claim_made("We cannot, and will not, see your password.", "we can see your password") is None


def test_clause_split_catches_claim_after_a_leading_no():
    assert claim_made("No, we can see your password.", "we can see your password")
    assert claim_made("We won't share it, but we can see your password.", "we can see your password")


def test_word_order_paraphrase_is_caught_but_synonyms_are_not():
    assert claim_made("Your password is something support can see.", "support can see your password")
    # Documented limitations: "view" is not "see", and irregular forms ("seen") are not stemmed.
    assert claim_made("Support can view your password.", "support can see your password") is None
    assert claim_made("Your password can be seen by support.", "support can see your password") is None


def test_must_include_matches_phrase_or_all_tokens_in_a_sentence():
    assert phrase_present("Use the Forgot Password link.", "forgot password")
    assert phrase_present("We email you a link to reset it.", "reset link")
    assert not phrase_present("Contact us.", "reset link")


def test_grounding_rewards_evidence_words_and_flags_unknown_numbers():
    ev = Evidence.build(EVIDENCE, "Can support see my password?")
    score, weak, bad_numbers, _ = grounding("Support staff cannot see your password.", ev, CFG)
    assert score == 1.0 and not weak and not bad_numbers
    score, weak, bad_numbers, _ = grounding("Your refund arrives within 24 hours by courier.", ev, CFG)
    assert score < 0.5 and weak and bad_numbers == ["24"]


def test_correct_refusal_is_not_marked_ungrounded():
    ev = Evidence.build(EVIDENCE, "What is the weather?")
    score, weak, _, all_refusal = grounding("I do not have that information. I can connect you with a human agent.", ev, CFG)
    assert all_refusal and score == 1.0 and not weak


def test_negated_certainty_marker_is_not_overconfident():
    q = _query(must_not_claim=[])
    row = check_answer(q, "v", "Support cannot guarantee a time. Use the reset link.", EVIDENCE, CFG)
    assert "overconfident_tone" not in row["failures"]
    row = check_answer(q, "v", "Use the reset link, it will definitely work.", EVIDENCE, CFG)
    assert "overconfident_tone" in row["risk_flags"]


def test_check_answer_row_shape_and_flags():
    row = check_answer(_query(), "prompt_x", "Support can see your password. Use the reset link.", EVIDENCE, CFG)
    assert set(row) == {"query_id", "variant", "retrieval_hit", "must_include_pass", "must_not_claim_pass",
                        "grounding_score", "risk_flags", "failures", "notes"}
    assert row["retrieval_hit"] and row["must_include_pass"] and not row["must_not_claim_pass"]
    assert "policy_violation" in row["risk_flags"]


def test_clean_answer_has_none_flag():
    row = check_answer(_query(), "v", "Support staff cannot see your password. Reset links expire after 30 minutes.", EVIDENCE, CFG)
    assert row["risk_flags"] == ["none"] and row["failures"] == []


def test_retrieval_miss_is_reported():
    row = check_answer(_query(expected_doc_ids=["D9"]), "v", "Support cannot see your password.", EVIDENCE, CFG)
    assert not row["retrieval_hit"] and "retrieval_miss" in row["failures"]
