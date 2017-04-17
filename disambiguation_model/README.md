### Specification

---

A classification model for disambiguation.

Input:

- Natural Language
- Random String

Output:

- Label, ( Whether the string pattern described in natural language occur in random string )

---

1. LSTM RNN encodes natural language
2. LSTM RNN ( character level ) encodes random string
3. Attention ? 
4. Feed Forward Neural Network