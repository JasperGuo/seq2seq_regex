## 1-layer Bidirectional Anaylyze Result

---

| Sentence  | Ground Truth  | Analysis  |
|---|---|---|
| lines with a capital letter before a character followed by a lower-case letter in them  |` ([<CAP>]).*(..*[<LOW>].*).* ` |  Almost Correct  | 
| items with a character preceding a numeral .  | `(..*[<NUM>].*)*`  | Ground Truth Error, Most prediction `(.)+([<NUM>]).*` which is actually **correct**  |  
| lines with 1 or more character , or ending with the string `<M0>` | `((.)+)|((.*)(<M0>))`  |  Half. Error examples: <br> `((.)+((.*)|(<M0>))(.*)` <br> `((.)|((.)|(<M0>)))*` |  
| lines with words numbers or letters | `\\b([<NUM>])|([<LET>])\\b` | Correct | 
| items with an upper case letter or vowel preceding a numeral . | `(([<VOW>])|([<CAP>])).*([<NUM>]).*` | Correct |
| lines ending with only a letter at least once | `((.*)([<LET>]))+` | Correct | 
| lines with the string `<M0>` or number before string `<M1>` | `(.*<M0>.*)|([<NUM>].*<M1>.*)` | **All incorrect** <br> Most predictions: <br> `((<M0>)|([<NUM>])).*(<M1>).*`  | 
| lines that contain a character followed by a letter , at least twice | `((.){2,}).*([<LET>]).*` | **All incorrect** <br> Most predictions:  `(..*[<LET>].*){2,}')`. <br> Ground truth doesn't make sense |
| lines with words with a character , vowel , or string `<M0>` | `\\b(.)|([<VOW>])|(<M0>)\\b` | Correct |
| lines containing a letter or string `<M0>`  | `.*([<LET>])|(<M0>).*` | Before Overfitting, Almost Correct |
| lines with the string `<M0>` containing only a lower-case letter | `((<M0>)|([<LOW>]))+` | Ground Truth Error, Sentence Description doesn not match the Ground truth | 
| lines which have the string `<M0>` followed by a number before the string `<M1>` | `(<M0>.*[<NUM>].*).*(<M1>).*` | Correct |
| lines containing the string `<M0>` followed by either the string `<M1>` , a letter , or a capital letter | `(<M0>).*((<M1>)|([<LET>])|([<CAP>])).*` | Almost Correct <br> Some Error examples: <br> `((<M0>)|([<LET>])|(<M1>)).*([<CAP>]).*` | 
| lines with a capital letter after a vowel at least zero times | `([<VOW>]).*(([<CAP>])*).*` | Almost Correct, but is unstable, which means it may be correct in this epoch, but it may be not in the next epoch  |
| lines ending with a number after a vowel | `((.*)([<VOW>])).*([<NUM>]).*` |   |
| lines containing a vowel or string `<M0>` | `.*([<VOW>])|(<M0>).*` | Almost correct, some error: <br> `\\b([<VOW>])|(<M0>)\\b` |
| lines with a lower-case letter before a character | `([<LOW>]).*(.*..*).*` | |
| lines containing both a lower-case and capital letter at least 3 times . | `([<LOW>].*[<CAP>].*){3,}` | **Almost incorrect** <br> Error examples: <br> `(.*[<LOW>].*){3,}`, <br> `([<LOW>]).*(([<CAP>]){3,}).*` <br> In ground truth, at least 3 three times, is modifing whole Sentence |
| lines containing a number after a vowel 5 or more times | `([<VOW>]).*(([<NUM>]){5,}).*` | **Almost incorrect** <br> what **5 or more times** does modify ? <br> Most Error Predictions: <br> `(([<VOW>]){5,}).*([<NUM>]).*` |
| lines with words with a character followed by a letter | `(\\b..*[<LET>].*\\b)+` | Correct |
| lines with a lower-case letter before `<M0>` | `([<LOW>]).*((.*<M0>.*)*).*` | Ground truth is incorrect, most predictions are `([<LOW>]).*((<M0>)+).*`, which should be correct |
| lines that contain a number followed by a character , at least 4 times | `([<NUM>]).*((.){4,}).*` | **Almost incorrect** <br> What **at least 4 times** should modify ? <br> Most predictions are: <br> `([<NUM>].*..*){4,}` |
| lines with string `<M0>` before letter , 2 or more times | `(<M0>).*(([<LET>]){2,}).*` | Correct |
| lines with words ending in string `<M0>` before lower-case letter | `\\b(.*)(<M0>.*[<LOW>].*)\\b` | Most Correct, Where is the word scope ? <br> Error Example: <br> `(\\b(.*)(<M0>)\\b).*([<LOW>]).*` |
| lines with the string `<M0>` before string `<M1>` followed by string `<M2>` | `(<M0>).*(<M1>.*<M2>.*).*` | Almost correct, but is unstable, which means it may be correct in this epoch, but it may be not in the next epoch |
| lines with with a letter followed by 2 or more of the string <M0> | `([<LET>].*<M0>.*){2,}` | Almost Correct | 
| lines followed by a vowel before a capital letter | `([<LOW>].*[<VOW>].*).*([<CAP>]).*` | Ground truth does't make sense |
| lines having 1 of the following: the string `<M0>` the string `<M1>` or a letter before the string `<M2>` | `((<M1>)|(<M1>)|([<LET>])).*(<M2>).*` | Ground Truth does not make sense, most predictions: <br> `((<M0>)|(<M1>)|([<LET>])).*(<M2>).*` |
| lines with lower-case letter before either character , vowel , or string `<M0>` | `([<LOW>]).*((.)|([<VOW>])|(<M0>)).*` | Correct |
| lines with the string <M0> followed by either a character or a letter | `(<M0>).*((.)|([<LET>])|([<VOW>])).*` | **Ground truth does not make sense, but most predictions match ground truth** |
| lines that have 4 or more numbers | `.*([<NUM>]){4,}.*` | **Almost incorrect**. Most Predictions: <br> `([<NUM>]){4,}` |
| lines with a character after a letter at least 5 times | `([<LET>]).*((.){5,}).*` | What **5 or more times** modifies ?  |
| lines starting with a lower-case letter with the string `<M0>` | `(([<LOW>])(.*)).*(<M0>).*` | It means **and** ? Most Correct |
| lines containing number before lower-case letter | `(.*[<NUM>].*).*([<LOW>]).*` | 
| lines starting with string `<M0>` or either a number , character , or letter | `((<M0>)|(([<NUM>])|(.)|([<LET>])))(.*)` | Correct, but is unstable, which means it may be correct in this epoch, but it may be not in the next epoch |
| lines ending with letter , zero or more times | `((.*)([<LET>]))*` | **Almost incorrect**. <br> What **zero or more times** modifies ? <br> most predictions: <br> `(.*)(([<LET>])*)`   | 
| lines which have a capital letter or a lower-case letter , 2 or more times | `(([<CAP>])|([<LOW>])){2,}` | Correct |
| lines containing 3 or more vowels before a lower-case letter | `(([<VOW>]){3,}).*([<LOW>]).*` | Correct |
| lines with words and zero or more of the string `<M0>` | `(\\b(<M0>)*\\b).*(.)+` | **Almost incorrect** What does the Sentence means ? |
| lines with a lower-case letter before a character | `(([<LET>])|([<LOW>])).*(.)+` | Ground truth does not make sense. Most predictions: <br> `([<LOW>]).*(..*..*).*')` or `([<LOW>]).*(.*..*).*` |
| lines with a letter before words with string `<M0>` | `([<LET>]).*(\\b.*<M0>.*\\b).*` | **Almost incorrect**, Most predictions: `([<LET>]).*(\\b(<M0>)+\\b).*` |
| lines with a lower-case letter preceded by either a vowel , the string <M0> or the string <M1> | `(([<VOW>])|(<M0>)|(<M1>)).*([<LOW>]).*` | Correct |
| lines starting in the string `<M0>` before a vowel | `((<M0>)(.*)).*(.*[<VOW>].*).*` | Correct | 
| lines with the string `<M0>` followed by a vowel , a capital letter , or a lower-case letter| `(<M0>).*(([<VOW>])|([<CAP>])|([<LOW>])).*` | Almost Correct |
| lines containing a vowel or string `<M0>` | `.*([<VOW>])|(<M0>).*` | Almost Correct |
| lines containing a vowel preceding the string <M0> before the string <M1> | `([<VOW>].*<M0>.*).*(<M1>).*` | Correct |
| lines with only zero or more letters | `(([<CAP>])|([<LET>]))*` | Ground truth does not make sense |
| lines with a vowel with a character | `(([<VOW>])|(.*..*))*` | What the Sentence means ? **and** ? <br> Most prediction: <br> `([<VOW>]).*((.)+).*` |
| lines having a capital letter before a number following a lower-case letter | `([<CAP>].*[<NUM>].*).*([<LOW>]).*` | Almost correct, but is unstable, which means it may be correct in this epoch, but it may be not in the next epoch |
| lines with the string `<M0>` before `<M1>` or `<M2>` or a lower-case letter | `(.*<M0>.*).*((<M1>)|(<M2>)|([<LOW>])).*` | **Almost incorrect** <br> Most Predictions: <br> `(<M0>).*((<M1>)|([<LOW>])|([<LOW>])).*` |
| lines with or without a vowel before a number | `(([<VOW>])*).*([<NUM>]).*` | Correct  |
| lines with words and `<M0>` or lower-case letter | `\b(<M0>)|(.*[<LOW>].*)\b` | **All incorrect** <br> Most predictions: <br> `\\b(<M0>)|([<LOW>])\\b`, What the sentence means ? |
| lines ending with a number | `(.*)(\b[<NUM>]\b)` | **All incorrect** <br> Most predictions: <br> `(.*)(.*[<NUM>].*)`, `(.*)(([<NUM>])+)` |
| lines with 2 or more numbers | `(.*[<NUM>].*){2,}` | **All incorrect** <br> Most predictions: <br> `([<NUM>]){2,}` |
