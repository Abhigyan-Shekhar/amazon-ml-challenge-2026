"""Document the 53-row manual review against the challenge's same-business rule."""
import csv
import json
from collections import Counter
from pathlib import Path

SHEET = Path("artifacts/validation/hgb_score_diagnostic/stratified_manual_review.csv")
RULE = "https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/PROBLEM_STATEMENT.md"

# A 'no' means the supplied ground-truth label appears inconsistent with the
# same-real-world-business rule. 'unclear' is deliberately used when names and
# addresses alone cannot establish identity.
DECISIONS = [
    ("no", "label_noise", "Unrelated names and different street numbers; shared street/city does not establish identity."),
    ("unclear", "other", "Names are close but 3746 versus 3732 Rome Drive may be a distinct business."),
    ("no", "ambiguous_shared_address", "Unrelated names at essentially the same unit; address alone cannot prove identity."),
    ("unclear", "other", "Same address and partial name, but 'Center' changes the business name."),
    ("unclear", "other", "Name core agrees, but 42 versus 32 Coburn Street is a material conflict."),
    ("yes", "other", "Best It name is a typo/legal-form variant and both addresses mention flat 604 in Pune."),
    ("yes", "missing_field_penalty", "Niketan Chemicals name agrees; target address is absent rather than conflicting."),
    ("unclear", "other", "Al Infra agrees, but added Center and shortened address leave identity uncertain."),
    ("yes", "transliteration_alias", "Hindi name transliterates to Life Software Private Limited; apartment clues agree."),
    ("yes", "other", "Personal-practice name and Brookdale address agree despite name and number noise."),
    ("yes", "other", "Akshay Trading name and flat 302 agree; target address is truncated."),
    ("yes", "transliteration_alias", "Odia name spells Bright Technologies Private Limited and plot/unit agree."),
    ("yes", "transliteration_alias", "Hindi name spells Best Solutions Pvt Ltd and the locality/address agree."),
    ("unclear", "missing_field_penalty", "Name core agrees, but the target omits address and has a Floating suffix."),
    ("yes", "transliteration_alias", "Gujarati name spells Galaxy Software and the office/complex address agrees."),
    ("yes", "other", "Setia Traders name core and detailed Golaghata address agree."),
    ("yes", "other", "Modi Electrical name and Rohan Iksha address agree despite number/script noise."),
    ("unclear", "other", "Bay Oil name agrees, but 5125 versus 5125-5127 could denote a neighboring unit."),
    ("yes", "missing_field_penalty", "Moyer Chemical name core agrees and target address is absent."),
    ("unclear", "other", "Ssb name and locality agree, but number 80 versus 8-0 needs verification."),
    ("yes", "missing_field_penalty", "Exact union name and number; target address is absent."),
    ("yes", "missing_field_penalty", "Exact Unico Software name apart from brackets; target address is absent."),
    ("unclear", "other", "Exact name, but 459 versus 59 Leedsville Road is a material address conflict."),
    ("yes", "other", "Highland Industries name and detailed building/address agree."),
    ("unclear", "ambiguous_shared_address", "Grace Synagogue versus Grace Center share an address but may be different organizations."),
    ("yes", "missing_field_penalty", "Hitech Hari Global name agrees; target address is absent."),
    ("no", "ambiguous_shared_address", "All First versus All-Services are different names sharing the same address."),
    ("yes", "other", "Frontier Federation name and Georgia Drive address agree despite accent/leading zero."),
    ("yes", "other", "Projectscholarshiphenderson.com plausibly encodes the same unique name; street address agrees."),
    ("yes", "other", "Distinctive ENT Associates name and apartment 17 agree despite omitted house number."),
    ("yes", "ambiguous_shared_address", "Different business names in the same complex/unit can be separate entities."),
    ("yes", "coincidental_overlap", "Name is close, but Clover versus Woodwick Street identifies different locations."),
    ("yes", "coincidental_overlap", "Maha Farming name overlaps, but 6511 versus 6522 is a material house-number change."),
    ("yes", "coincidental_overlap", "Agro Maa core overlaps, but Inventure versus Technocrats and address detail differ."),
    ("yes", "coincidental_overlap", "Orthopedic Partners words overlap, but Indiana versus Utah is a clear geographic conflict."),
    ("yes", "coincidental_overlap", "Same generic dentistry name and unit, but 5010 versus 5021 indicates different premises."),
    ("yes", "coincidental_overlap", "Rishi name token and number overlap, but full names and cities differ."),
    ("unclear", "missing_field_penalty", "Name is identical after word order, but no target address is available to disambiguate."),
    ("unclear", "ambiguous_shared_address", "Khatu Group may be related, but Enterprises and shop 3 versus 8 suggest distinct units."),
    ("yes", "coincidental_overlap", "Drayon Commercial name agrees, but 278 versus 281 is a distinct street number."),
    ("yes", "ambiguous_shared_address", "Unrelated company names share a detailed office address."),
    ("unclear", "other", "Swarajya Sangh Care name and unit agree; Phase-1A versus Phase-12A may be a typo, so label noise is unproven."),
    ("yes", "coincidental_overlap", "Identical generic medical name but entirely different street addresses."),
    ("yes", "coincidental_overlap", "Dental Specialists name overlaps, but 3308 versus 3315 differs."),
    ("yes", "coincidental_overlap", "Raj Logistics name agrees but Varanasi versus Howrah clearly differs."),
    ("yes", "coincidental_overlap", "Safex Welfare Society name core agrees but house 415 versus 428 differs."),
    ("yes", "coincidental_overlap", "Sterling Crystal Drive name agrees but 2062 versus 2073 differs."),
    ("yes", "ambiguous_shared_address", "Different company names share Vaishnavi Tech Park address."),
    ("yes", "coincidental_overlap", "Pelium name core and unit agree, but 2420 versus 2425 differs."),
    ("yes", "coincidental_overlap", "Dentist name agrees but 2275 versus 2288 Stream Vista Place differs."),
    ("yes", "ambiguous_shared_address", "Rana Cosmetics and Sunrise Media are different names at the same unit."),
    ("unclear", "ambiguous_shared_address", "Dreaming Hospital core agrees, but B-1 versus B-2 and Overseas may denote another entity."),
    ("yes", "ambiguous_shared_address", "Agasti Brothers and Blue Exports have different names despite similar street address."),
]

# For every missed-pair "other", record the observed failure mode and why the
# five named categories do not fit. Row numbers refer to the original CSV order.
OTHER_DETAILS = {
    2: ("house_number_conflict", "3746 versus 3732 is a material address conflict; neither address is missing and the name is not a semantic alias."),
    4: ("name_variant_ambiguous", "Center may change the entity identity; the shared address alone cannot validate a semantic alias."),
    5: ("house_number_conflict", "42 versus 32 is a material address conflict; the name core is already close."),
    6: ("legal_suffix_and_typo", "Private versus Privte and M/s are form/typo noise, not transliteration or a semantic alias."),
    8: ("name_variant_ambiguous", "The added Center makes identity unclear; the address is shortened but present."),
    10: ("ocr_typo", "Jccl,s is an OCR-like corruption of Jones; the address is present and no alias is established."),
    11: ("partial_address", "The target retains flat 302 but drops street context; the address is partial, not missing."),
    16: ("title_and_legal_form", "Dr and Ltd are extra name tokens; the distinctive Setia Traders core is unchanged."),
    17: ("number_and_script_noise", "Rohan Iksha and name core agree; the numeric formatting and Kannada state token are the gap."),
    18: ("address_range_ambiguity", "5125 versus 5125-5127 may be a range or neighboring unit; the name core is close."),
    20: ("numeric_segmentation", "80 versus 8-0 may be number formatting; both addresses are present."),
    23: ("house_number_conflict", "459 versus 59 is a material conflict even though the names are essentially exact."),
    24: ("honorific_and_legal_form", "Shri and Pvt Ltd are title/legal-form changes; detailed address agrees."),
    28: ("accent_and_leading_zero", "The name differs by an accent and address by a leading zero, not a semantic alias."),
    29: ("website_name_alias", "The web-domain form concatenates the unique name; this is a valid lexical-to-semantic alias."),
    30: ("organizational_designator", "Federation is an added organization word; distinctive ENT name and apartment agree."),
}

SEMANTIC_ALIAS_ROWS = {29, 30}
LEXICAL_VARIANT_ROWS = {6, 10, 16, 24, 28, 29, 30}
OTHER_GROUPS = {
    "address_number_conflict": {2, 5, 18, 20, 23},
    "name_identity_ambiguous": {4, 8},
    "name_form_or_ocr": {6, 10, 16, 24, 28, 30},
    "partial_address": {11},
    "mixed_address_script": {17},
    "semantic_website_alias": {29},
}


def main():
    with SHEET.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert len(rows) == len(DECISIONS) == 53
    assert all(row["case_type"] == ("false_negative_true_pair" if i < 30 else "false_positive") for i, row in enumerate(rows))
    for row, (valid, category, notes) in zip(rows, DECISIONS):
        row.update(label_valid_under_spec=valid, error_category=category, notes=notes)
    assert {i for i, row in enumerate(rows, 1) if row["case_type"] == "false_negative_true_pair" and row["error_category"] == "other"} == set(OTHER_DETAILS)
    assert set().union(*OTHER_GROUPS.values()) == set(OTHER_DETAILS)
    for i, row in enumerate(rows, 1):
        subtype, reason = OTHER_DETAILS.get(i, ("", ""))
        group = next((name for name, indices in OTHER_GROUPS.items() if i in indices), "")
        row.update(other_subtype=subtype, other_group=group, other_exclusion_reason=reason)
    annotation_fields = ["label_valid_under_spec", "error_category", "notes"]
    annotation_fields += ["other_subtype", "other_group", "other_exclusion_reason"]
    output_fields = fields + [field for field in annotation_fields if field not in fields]
    with SHEET.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    count = {case: dict(Counter(row["error_category"] for row in rows if row["case_type"] == case)) for case in ("false_negative_true_pair", "false_positive")}
    missed = rows[:30]
    semantic_valid = sum(missed[i-1]["label_valid_under_spec"] == "yes" for i in SEMANTIC_ALIAS_ROWS)
    lexical_variants = sum(missed[i-1]["label_valid_under_spec"] == "yes" for i in LEXICAL_VARIANT_ROWS)
    alias = count["false_negative_true_pair"].get("transliteration_alias", 0)
    ceiling = sum(row["error_category"] in ("label_noise", "ambiguous_shared_address") for row in missed)
    unresolved_other = sum(row["label_valid_under_spec"] == "unclear" for row in missed if row["error_category"] == "other")
    observed_embedding = alias + semantic_valid
    upper_embedding = observed_embedding + unresolved_other
    report = {"linkage_definition": "same real-world business entity, not merely same address or lexical overlap", "source": RULE, "counts": count, "other_groups": dict(Counter(row["other_group"] for row in missed if row["error_category"] == "other")), "other_label_valid_counts": dict(Counter(row["label_valid_under_spec"] for row in missed if row["error_category"] == "other")), "other_subtypes": dict(Counter(row["other_subtype"] for row in missed if row["error_category"] == "other")), "missed_alias_plus_semantic_valid": observed_embedding, "missed_alias_plus_semantic_valid_share": observed_embedding / 30, "missed_embedding_upper_bound_if_unclear_other_count": upper_embedding, "missed_embedding_upper_bound_if_unclear_other_share": upper_embedding / 30, "missed_broad_name_lexical_gap": alias + lexical_variants, "missed_broad_name_lexical_gap_share": (alias + lexical_variants) / 30, "missed_label_noise_plus_ambiguous_address": ceiling, "missed_label_noise_plus_ambiguous_address_share": ceiling / 30, "gate_embedding_cleared": observed_embedding / 30 >= .4, "embedding_gate_status": "cleared" if observed_embedding / 30 >= .4 else "indeterminate" if upper_embedding / 30 >= .4 else "not_cleared", "gate_ceiling_dominates": ceiling / 30 >= .5}
    target = SHEET.with_name("annotation_gate.json")
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
