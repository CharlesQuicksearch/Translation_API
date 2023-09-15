import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model = AutoModelForSeq2SeqLM.from_pretrained("Model/Translation_nllb_Model")
_tokenizer = AutoTokenizer.from_pretrained("Model/Translation_nllb_Tokenizer")

language_codes = np.array(
        [("arb_Arab", "ar-sa", "ar"),
         ("eng_Latn", 'en-AU', 'en-US'),
         ("eng_Latn", 'en-GB', 'en'),
         ("swe_Latn", 'sv-SE', 'sv'),
         ("cat_Latn", "ca-ES", "ca"),
         ("zho_Hans", "zh-CN", "zh"),
         ("zho_Hans", "zh-Hant", "zh"),
         ("ces_Latn", "cs-CZ", "cz"),
         ("nld_Latn", "nl-BE","nl"), #nl added not in list
         ("est_Latn", "et-EE", "et"),
         ("fra_Latn", "fr-FR", "fr"),
         ("deu_Latn", "de-AT", "de"),
         ("ell_Grek", "el-GR", "el"),
         ("ind_Latn", "id", "id"),
         ("ita_Latn", "it-IT", "it"),
         ("jpn_Jpan", "ja-JP", "ja"),
         ("lvs_Latn", "lv-LV", "lv"),
         ("pol_Latn", "pl-PL", "pl"),
         ("por_Latn", "pt-PT", "pt"),
         ("ron_Latn", "ro-RO", "ro"),
         ("rus_Cyrl", "ru-RU", "ru"),
         ("ukr_Cyrl", "uk", "uk"), #Ukrainian
         ("spa_Latn", "es-ES", "es"),
         ("tur_Latn", "tr-TR", "tr")],
        dtype=[('id', 'U10'), ('code1', 'U10'), ('code2', 'U10')])

async def translate(input, src_lang, tgt_lang):

    src_code = get_model_code(src_lang)
    tgt_code = get_model_code(tgt_lang)

    if src_code is None or tgt_code is None:


        return f"Error incorrect code. src_lang: {src_code}, tgt_code: {tgt_code}"

    translator = pipeline('translation', model=model, tokenizer=_tokenizer, src_lang=src_code, tgt_lang=tgt_code, max_length = 400)

    translated_review = translator(input)

    return translated_review[0]['translation_text']

def get_model_code(lang_code):
    if len(lang_code) < 3:
        matching_row = language_codes[language_codes['code2'] == lang_code]
        return matching_row['id'][0].item()
    if len(lang_code) > 2:
        matching_row = language_codes[language_codes['code1'] == lang_code]
        return matching_row['id'][0].item()
    else:
        return None
