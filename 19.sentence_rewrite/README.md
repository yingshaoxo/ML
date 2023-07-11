# Sentence Rewrite (Text Formalization)

## Method 1
* Use 'Sentence correction' model to get a correct sentence.
    > https://huggingface.co/shibing624/mengzi-t5-base-chinese-correction
* Use 'Punctuation correction' model to get a more correct sentence.
    > https://huggingface.co/raynardj/classical-chinese-punctuation-guwen-biaodian

> If you have an advanced AI bot, you may directly ask him/she "Do you know how to speak 'xxx' in a another better way?"

## Method 2
* Use 'Punctuation and grammar correction' model to get a more correct sentence.
    > https://huggingface.co/oliverguhr/spelling-correction-english-base
* Use 'BART Paraphrase' model to get a simpler sentence.
    > https://huggingface.co/eugenesiow/bart-paraphrase
* [Optional] Use 'sentence style transfering' or 'text rewrite' or 'Text Formalizing' model to rewrite the sentence.
    > https://huggingface.co/tuner007/pegasus_paraphrase

## Method 3
* Use 'Offline Google translation' model to translate that sentence to English language.
* Use 'Punctuation and grammar correction' model to get a more correct sentence.
    > https://huggingface.co/oliverguhr/spelling-correction-english-base
* [Optional] Use 'sentence style transfering' or 'text rewrite' or 'Text Formalizing' model to rewrite the sentence.
    > https://huggingface.co/tuner007/pegasus_paraphrase
* Translate English sentence back to the original language.

> If you have an advanced AI bot, you may directly ask him/she "Do you know how to speak 'xxx' in a another better way?"

## Author 
yingshaoxo
