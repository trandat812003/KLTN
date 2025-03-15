
_ESCONV_QUESTION = "[Question]"
_ESCONV_RESTATEMENT = "[Restatement or Paraphrasing]"
_ESCONV_REFLECTION = "[Reflection of feelings]"
_ESCONV_SELF_DISCLOSURE = "[Self-disclosure]"
_ESCONV_AFFIRMATION = "[Affirmation and Reassurance]"
_ESCONV_SUGGESTION = "[Providing Suggestions]"
_ESCONV_INFORMATION = "[Information]"
_ESCONV_OTHERS = "[Others]"


_MI_GIVING_INFORMATION = "[GIV]"
_MI_QUESTION = "[QUEST]"
_MI_SEEKING_INFORMATION = "[SEEK]"
_MI_AFFIRMATION = "[AF]"
_MI_PROVIDING_WITH_PARAPHRASING = "[PWP]"
_MI_PROVIDING_WITHOUT_PARAPHRASING = "[PWOP]"
_MI_EMPHASIS = "[EMPH]"
_MI_CONTEXTUALIZATION = "[CON]"
_MI_SELF_REFLECTION = "[SR]"
_MI_CLARIFICATION_REQUEST = "[CR]"


labels_dataset_mapping = {
    "mi": {
        ("low", "low", "low"): _MI_SEEKING_INFORMATION,
        ("low", "low", "medium"): _MI_CLARIFICATION_REQUEST,
        ("low", "low", "high"): _MI_SELF_REFLECTION,
        ("low", "medium", "low"): _MI_CONTEXTUALIZATION,
        ("low", "medium", "medium"): _MI_GIVING_INFORMATION,
        ("low", "medium", "high"): _MI_PROVIDING_WITHOUT_PARAPHRASING,
        ("low", "high", "low"): _MI_QUESTION,
        ("low", "high", "medium"): _MI_AFFIRMATION,
        ("low", "high", "high"): _MI_PROVIDING_WITHOUT_PARAPHRASING,
        ("medium", "low", "low"): _MI_CLARIFICATION_REQUEST,
        ("medium", "low", "medium"): _MI_SEEKING_INFORMATION,
        ("medium", "low", "high"): _MI_AFFIRMATION,
        ("medium", "medium", "low"): _MI_CONTEXTUALIZATION,
        ("medium", "medium", "medium"): _MI_GIVING_INFORMATION,
        ("medium", "medium", "high"): _MI_PROVIDING_WITH_PARAPHRASING,
        ("medium", "high", "low"): _MI_QUESTION,
        ("medium", "high", "medium"): _MI_AFFIRMATION,
        ("medium", "high", "high"): _MI_PROVIDING_WITHOUT_PARAPHRASING,
        ("high", "low", "low"): _MI_CLARIFICATION_REQUEST,
        ("high", "low", "medium"): _MI_SEEKING_INFORMATION,
        ("high", "low", "high"): _MI_AFFIRMATION,
        ("high", "medium", "low"): _MI_CONTEXTUALIZATION,
        ("high", "medium", "medium"): _MI_GIVING_INFORMATION,
        ("high", "medium", "high"): _MI_PROVIDING_WITH_PARAPHRASING,
        ("high", "high", "low"): _MI_QUESTION,
        ("high", "high", "medium"): _MI_EMPHASIS,
        ("high", "high", "high"): _MI_PROVIDING_WITHOUT_PARAPHRASING,
    },

    "esconv": {
        ("low", "low", "low"): _ESCONV_QUESTION,
        ("low", "low", "medium"): _ESCONV_RESTATEMENT,
        ("low", "low", "high"): _ESCONV_REFLECTION,
        ("low", "medium", "low"): _ESCONV_SELF_DISCLOSURE,
        ("low", "medium", "medium"): _ESCONV_AFFIRMATION,
        ("low", "medium", "high"): _ESCONV_SUGGESTION,
        ("low", "high", "low"): _ESCONV_INFORMATION,
        ("low", "high", "medium"): _ESCONV_OTHERS,
        ("low", "high", "high"): _ESCONV_SUGGESTION,
        ("medium", "low", "low"): _ESCONV_RESTATEMENT,
        ("medium", "low", "medium"): _ESCONV_QUESTION,
        ("medium", "low", "high"): _ESCONV_AFFIRMATION,
        ("medium", "medium", "low"): _ESCONV_SELF_DISCLOSURE,
        ("medium", "medium", "medium"): _ESCONV_AFFIRMATION,
        ("medium", "medium", "high"): _ESCONV_SUGGESTION,
        ("medium", "high", "low"): _ESCONV_INFORMATION,
        ("medium", "high", "medium"): _ESCONV_AFFIRMATION,
        ("medium", "high", "high"): _ESCONV_SUGGESTION,
        ("high", "low", "low"): _ESCONV_RESTATEMENT,
        ("high", "low", "medium"): _ESCONV_QUESTION,
        ("high", "low", "high"): _ESCONV_AFFIRMATION,
        ("high", "medium", "low"): _ESCONV_SELF_DISCLOSURE,
        ("high", "medium", "medium"): _ESCONV_AFFIRMATION,
        ("high", "medium", "high"): _ESCONV_SUGGESTION,
        ("high", "high", "low"): _ESCONV_INFORMATION,
        ("high", "high", "medium"): _ESCONV_OTHERS,
        ("high", "high", "high"): _ESCONV_SUGGESTION,
    }
}
