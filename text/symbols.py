""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

## from text import cmudict, pinyin
from text import pinyin

_pad = "_"
_punctuation = "!'(),.:;? " ## 10
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" ## 52
### added @ pinyin.py or nots
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ["@" + s for s in cmudict.valid_symbols] ## 84
_pinyin = ["@" + s for s in pinyin.valid_symbols] ## 209

_pinyin6 = [s for s in pinyin.valid_symbols6] #  [ENGLISH]
_pinyin5 = [s for s in pinyin.valid_symbols5] #  [FRENCH]
_pinyin4 = [s for s in pinyin.valid_symbols4] #  [ENGLISH]
_pinyin3 = [s for s in pinyin.valid_symbols3] #  [GERMAN]
_pinyin2 = [s for s in pinyin.valid_symbols2] #  [GERMAN]

_pinyin2_b = [s for s in pinyin.valid_symbols2_b]

### T1
# Export all symbols:
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     + _arpabet
#     + _pinyin
#     + _pinyin2_b
#     # + list(_letters_ipa) ## added @ pinyin.py or not
#     + _silences
# )

### T2
# Export all symbols:
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     + _arpabet
#     + _pinyin
#     + _pinyin2
#     # + list(_letters_ipa) ## added @ pinyin.py or not
#     + _silences
# )

### T3
# Export all symbols:
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     # + _arpabet
#     # + _pinyin
#     + _pinyin3
#     # + list(_letters_ipa) ## added @ pinyin.py or not
#     + _silences
# )

### T4_MR
# Export all symbols:
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     # + list(_letters)
#     # + _arpabet
#     # + _pinyin
#     + _pinyin4
#     # + list(_letters_ipa) ## added @ pinyin.py or not
#     + _silences
# )


### T5 [FRENCH]
# Export all symbols:
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     # + _arpabet
#     # + _pinyin
#     + _pinyin5
#     # + list(_letters_ipa) ## added @ pinyin.py or not
#     + _silences
# )


### T6 [ENGLISH]
# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    # + list(_letters)
    # + _arpabet
    # + _pinyin
    + _pinyin6
    # + list(_letters_ipa) ## added @ pinyin.py or not
    + _silences
)
