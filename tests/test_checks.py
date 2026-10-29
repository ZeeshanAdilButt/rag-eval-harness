from rag_eval.checks import check_answer, claim_made, grounding, Evidence, phrase_present
from rag_eval.config import ChecksConfig, load_config
from rag_eval.schema import Doc, Query
from rag_eval.text import clauses, normalize, sentences, stem

CFG = load_config(None).checks.model_copy(update={
    "certainty_markers": ["guarantee", "guaranteed", "definitely"],
    "refusal_markers": ["human agent", "i do not have"],
})
EVIDENCE = [{"doc_id": "D1", "score": 2.0, "title": "Password reset", "text": "Support staff cannot see your password. Reset links expire after 30 minutes."}]


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


KB = {"D1": Doc(doc_id="D1", title="Password reset", text=EVIDENCE[0]["text"])}
OTHER = [{"doc_id": "D2", "score": 1.0, "title": "Fees", "text": "Withdrawal fees are listed in the pricing table."}]


def test_retrieval_miss_grades_against_expected_docs_from_kb():
    answer = "Support staff cannot see your password. Reset links expire after 30 minutes."
    row = check_answer(_query(), "v", answer, OTHER, CFG)
    assert "unsupported_claim" in row["failures"]  # without the KB the correct answer looks invented
    row = check_answer(_query(), "v", answer, OTHER, CFG, KB)
    assert "retrieval_miss" in row["failures"] and "unsupported_claim" not in row["failures"]
    assert "graded against expected docs because retrieval missed" in row["notes"]
    assert "graded against" not in check_answer(_query(), "v", answer, EVIDENCE, CFG, KB)["notes"]


def test_zero_score_expected_doc_is_not_a_hit():
    row = check_answer(_query(), "v", "Support cannot see your password.", [{**EVIDENCE[0], "score": 0.0}], CFG)
    assert not row["retrieval_hit"] and "retrieval_miss" in row["failures"]


def test_no_expected_docs_is_not_a_miss():
    row = check_answer(_query(expected_doc_ids=[]), "v", "I do not have that. Ask a human agent.", EVIDENCE, CFG)
    assert row["retrieval_hit"] and "retrieval_miss" not in row["failures"] and "no expected docs" in row["notes"]


def test_conjunction_splits_clauses_and_negation_is_scoped():
    assert clauses("There is no delay and your withdrawal will be approved today") == [
        "There is no delay", "your withdrawal will be approved today"]
    claim = "your withdrawal will be approved"
    assert claim_made("There is no delay and your withdrawal will be approved today.", claim)
    assert claim_made("Your withdrawal will not be approved today because of the review.", claim) is None
    # "without" only reaches its object, so the verb after it is still asserted.
    assert claim_made("Without delay we approve your withdrawal.", "we approve your withdrawal")
    assert claim_made("We cannot approve your withdrawal without a review.", "we approve your withdrawal") is None
    # Documented limit: "no" negates anything within 3 tokens, so "No delay, we approve it" style
    # sentences without a split word read as negated.
    assert claim_made("There is no delay we approve your withdrawal.", "we approve your withdrawal") is None


def test_must_include_has_token_boundaries_and_respects_negation():
    assert not phrase_present("I know the answer.", "no")
    assert not phrase_present("You can not reset your password.", "reset password")
    assert phrase_present("You can reset your password.", "reset password")
    assert phrase_present("Demo funds have no cash value.", "no cash value")
    assert not phrase_present("Demo funds have a cash value.", "no cash value")


def test_list_indices_are_not_unsupported_numbers():
    ev = Evidence.build(EVIDENCE, "Can support see my password?")
    assert sentences("1. Reset links expire.\n2) Support cannot see it.") == ["Reset links expire.", "Support cannot see it."]
    _, _, bad, _ = grounding("1. Reset links expire after 30 minutes.\n2) Support staff cannot see your password.", ev, CFG)
    assert bad == []


def test_percent_is_kept_so_it_meets_a_marker():
    assert normalize("100% safe") == "100 percent safe"
    cfg = CFG.model_copy(update={"certainty_markers": ["100 percent"]})
    row = check_answer(_query(must_not_claim=[]), "v", "Your reset link is 100% safe.", EVIDENCE, cfg)
    assert "overconfident_tone" in row["failures"]


def test_marker_present_in_evidence_is_not_overconfident():
    ev_docs = [{**EVIDENCE[0], "text": EVIDENCE[0]["text"] + " We guarantee reset links work."}]
    q = _query(must_not_claim=[])
    assert "overconfident_tone" not in check_answer(q, "v", "We guarantee the reset link works.", ev_docs, CFG)["failures"]
    assert "overconfident_tone" in check_answer(q, "v", "We guarantee the reset link works.", EVIDENCE, CFG)["failures"]


def test_refusal_clause_does_not_exempt_the_rest_of_the_sentence():
    ev = Evidence.build(EVIDENCE, "Can support see my password?")
    _, _, bad, all_refusal = grounding("Please contact support, your refund is approved for 3 days", ev, CFG.model_copy(
        update={"refusal_markers": ["contact support"]}))
    assert bad == ["3"] and not all_refusal


def test_stemmer_joins_plurals_and_keeps_double_s():
    assert stem("process") == stem("processes")
    assert stem("policies") == stem("policy")
    assert stem("address") == stem("addresses")


def test_stemmer_joins_ification_and_ied_forms():
    assert stem("verification") == stem("verified") == stem("verify")
    assert claim_made("You may withdraw after your account is upgraded and verified.", "yes, after verification")


def test_one_word_claim_is_negated_by_its_sentence():
    claim = "yes, after verification"
    assert claim_made("No. Demo funds use virtual money, so you can't withdraw, even after verification.", claim) is None
    assert claim_made("You can withdraw after verification.", claim)
