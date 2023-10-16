

############################################################################################################
#  T4_MR [GERMAN]: cmudict, pinyin.valid_symbols, m_adds, ... (except ger_totals) all phones are excluded  #
############################################################################################################
## https://mfa-models.readthedocs.io/en/latest/dictionary/German/German%20MFA%20dictionary%20v2_0_0a.html
mfa_official_ger = "a aj aw aː b c cʰ d eː f h iː j k kʰ l l̩ m m̩ n n̩ oː p pf pʰ s t ts tʃ tʰ uː v x yː z ç øː ŋ œ ɐ ɔ ɔʏ ə ɛ ɟ ɡ ɪ ɲ ʁ ʃ ʊ ʏ"
mfa_ger = mfa_official_ger.split(" ")
mfa_ger = sorted(['@'+s for s in mfa_ger if s != ' '])
valid_symbols4 = mfa_ger ## n_vocab: 52 + 15 = 67


#################
#  T6 [ENGLISH] #
#################
# https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20MFA%20dictionary%20v2_2_1.html
mfa_official_ens_221 = "a aj aw aː b bʲ c cʰ cʷ d dʒ dʲ d̪ e ej f fʲ fʷ h i iː j k kp kʰ kʷ l m mʲ m̩ n n̩ o ow p pʰ pʲ pʷ s t tʃ tʰ tʲ tʷ t̪ u uː v vʲ vʷ w z æ ç ð ŋ ɐ ɑ ɑː ɒ ɒː ɔ ɔj ə əw ɚ ɛ ɛː ɜ ɜː ɝ ɟ ɟʷ ɡ ɡb ɡʷ ɪ ɫ ɫ̩ ɲ ɹ ɾ ɾʲ ɾ̃ ʃ ʉ ʉː ʊ ʎ ʒ ʔ θ"
mfa_ens221 = sorted(['@'+s for s in mfa_official_ens_221.split(" ") if s != ' '])
valid_symbols6 = mfa_ens221 ## n_vocab: 92 + 15 = 107

#################
#  T7 [KOREAN] #
#################
# https://mfa-models.readthedocs.io/en/latest/dictionary/Korean/Korean%20MFA%20dictionary%20v2_0_0a.html
mfa_official_kr = 'b d dʑ e eː h i iː j k kʰ k̚ k͈ m n o oː p pʰ p̚ p͈ s sʰ s͈ t tɕ tɕʰ tɕ͈ tʰ t̚ t͈ u uː w x ç ŋ ɐ ɕʰ ɕ͈ ɛ ɛː ɡ ɣ ɥ ɦ ɨ ɨː ɭ ɰ ɲ ɸ ɾ ʌ ʌː ʎ ʝ β'
mfa_kr200a= sorted(['@' + s for s in mfa_official_kr.split(' ') if s != ' ']) 
valid_symbols7 = mfa_kr200a ## n_vocab: 58 + 15 = 73
