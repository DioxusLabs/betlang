# Actual Dataset Confusion Matrix

Checkpoint: `assets/magika/source-student-q4.bin`
Source export: `assets/magika/source-student-q4-guesslang-direct-fs-pruned48-qate1.bin`
Confusion JSON: `assets/magika/source-student-q4-guesslang-direct-fs-pruned48-qate1-confusion.json`

Labels: 48 active filesystem-backed labels. Removed unsupported labels include `jsonl`, `matlab`, and `prolog`.
Test rows: 182,700. Correct: 176,392. Test fs accuracy / micro recall: 0.965473. Macro recall: 0.965813.

## Worst Per-Class Recall

| Label | Correct | Total | Recall |
|---|---:|---:|---:|
| cpp | 2,804 | 3,356 | 0.835518 |
| ini | 979 | 1,083 | 0.903970 |
| c | 5,797 | 6,334 | 0.915219 |
| shell | 3,501 | 3,791 | 0.923503 |
| cobol | 185 | 198 | 0.934343 |
| groovy | 1,722 | 1,841 | 0.935361 |
| javascript | 7,215 | 7,688 | 0.938476 |
| batch | 3,767 | 3,981 | 0.946245 |
| php | 3,802 | 3,988 | 0.953360 |
| asm | 3,783 | 3,968 | 0.953377 |
| julia | 3,815 | 4,000 | 0.953750 |
| powershell | 3,819 | 3,998 | 0.955228 |
| markdown | 3,782 | 3,954 | 0.956500 |
| perl | 7,431 | 7,752 | 0.958591 |
| objectivec | 6,038 | 6,279 | 0.961618 |

## Top Confusions

| Actual | Predicted | Count | Actual total | Actual recall |
|---|---|---:|---:|---:|
| cpp | c | 449 | 3,356 | 0.835518 |
| c | cpp | 403 | 6,334 | 0.915219 |
| javascript | typescript | 198 | 7,688 | 0.938476 |
| shell | batch | 148 | 3,791 | 0.923503 |
| php | html | 131 | 3,988 | 0.953360 |
| toml | ini | 107 | 3,323 | 0.962684 |
| batch | shell | 81 | 3,981 | 0.946245 |
| typescript | javascript | 76 | 3,999 | 0.963241 |
| perl | r | 76 | 7,752 | 0.958591 |
| powershell | batch | 67 | 3,998 | 0.955228 |
| javascript | yaml | 64 | 7,688 | 0.938476 |
| objectivec | cpp | 62 | 6,279 | 0.961618 |
| scala | kotlin | 57 | 4,000 | 0.970750 |
| dockerfile | shell | 57 | 3,781 | 0.979106 |
| perl | objectivec | 51 | 7,752 | 0.958591 |
| objectivec | c | 47 | 6,279 | 0.961618 |
| shell | powershell | 45 | 3,791 | 0.923503 |
| perl | erlang | 40 | 7,752 | 0.958591 |
| groovy | java | 39 | 1,841 | 0.935361 |
| typescript | xml | 37 | 3,999 | 0.963241 |
| javascript | json | 37 | 7,688 | 0.938476 |
| scala | groovy | 36 | 4,000 | 0.970750 |
| julia | r | 34 | 4,000 | 0.953750 |
| cs | java | 34 | 3,999 | 0.984746 |
| c | objectivec | 33 | 6,334 | 0.915219 |
| lua | javascript | 32 | 3,972 | 0.962739 |
| objectivec | perl | 31 | 6,279 | 0.961618 |
| julia | ruby | 30 | 4,000 | 0.953750 |
| clojure | lisp | 30 | 3,941 | 0.986044 |
| kotlin | gradle | 29 | 4,000 | 0.980750 |
| batch | ini | 29 | 3,981 | 0.946245 |
| asm | ini | 29 | 3,968 | 0.953377 |
| asm | c | 29 | 3,968 | 0.953377 |
| html | xml | 27 | 3,971 | 0.966759 |
| cmake | shell | 27 | 7,456 | 0.976529 |
| shell | markdown | 26 | 3,791 | 0.923503 |
| javascript | markdown | 26 | 7,688 | 0.938476 |
| html | markdown | 26 | 3,971 | 0.966759 |
| batch | powershell | 26 | 3,981 | 0.946245 |
| asm | batch | 26 | 3,968 | 0.953377 |
