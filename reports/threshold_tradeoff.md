# How the 0/1 column is cut (out-of-fold, test-sized 4,000-row chunks)

| how the 0/1 column is cut                     |   flagged |   precision |   recall |     F1 |   PR-AUC of 0/1 column |   mean threshold |
|:----------------------------------------------|----------:|------------:|---------:|-------:|-----------------------:|-----------------:|
| plug-in: max expected binary PR-AUC (default) |    0.0067 |      0.5124 |   0.1936 | 0.2796 |                 0.1154 |           0.2833 |
| plug-in: max expected F1                      |    0.0118 |      0.3709 |   0.2484 | 0.2969 |                 0.1078 |           0.1586 |
| fixed threshold 0.095                         |    0.0198 |      0.2683 |   0.3022 | 0.2836 |                 0.0953 |           0.095  |
| fixed threshold 0.16                          |    0.0115 |      0.3777 |   0.2474 | 0.2982 |                 0.1088 |           0.16   |
| fixed threshold 0.25                          |    0.0072 |      0.4889 |   0.2011 | 0.2842 |                 0.1147 |           0.25   |
| fixed threshold 0.3                           |    0.006  |      0.5455 |   0.186  | 0.2769 |                 0.1179 |           0.3    |
| fixed threshold 0.4                           |    0.0047 |      0.608  |   0.1615 | 0.2544 |                 0.1163 |           0.4    |
