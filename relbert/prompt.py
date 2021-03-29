""" Prompting function """

preset_templates = {
    "a": "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>",
    "b": "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is A's <mask>",
    "c": "Today, I finally discovered the relation between <subj> and <obj> : <mask>",
    "d": "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>",
    "e": "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is A’s <mask>",
    "f": "The teacher explained how <subj> is related to <obj> : it is <obj> 's <mask>",
    "g": "The teacher explained how <subj> is related to <obj> : it is <mask>",
    "h": "The teacher explained how <subj> is related to <obj> : <mask>",
    "i": "The teacher explained how <subj> is related to <obj> : it is the <mask>"
}

token_mask = '<mask>'
token_subject = '<subj>'
token_object = '<obj>'

__all__ = ('token_object', 'token_subject', 'token_mask', 'preset_templates', 'word_pair_prompter')


def word_pair_prompter(word_pair, template_type: str = 'a', custom_template: str = None, mask_token: str = None):
    """ Transform word pair into string prompt. """
    if custom_template is not None:
        assert token_mask in custom_template, 'mask token not found: {}'.format(custom_template)
        assert token_subject in custom_template, 'subject token not found: {}'.format(custom_template)
        assert token_object in custom_template, 'object token not found: {}'.format(custom_template)
        template = custom_template
    else:
        template = preset_templates[template_type]

    assert len(word_pair) == 2, word_pair
    subj, obj = word_pair
    assert token_subject not in subj and token_object not in subj and token_mask not in subj
    assert token_subject not in obj and token_object not in obj and token_mask not in obj
    prompt = template.replace(token_subject, subj).replace(token_object, obj)
    if mask_token is not None:
        prompt = prompt.replace(token_mask, mask_token)
    return prompt
