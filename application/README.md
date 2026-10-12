# Empirical application datasets

The datasets cover listed firms in two Chinese manufacturing industries in 2024. Data were obtained from the China Stock Market & Accounting Research database (CSMAR).

| File | Industry | Firms |
|---|---|---:|
| [CSMAR_C27_2024.csv](CSMAR_C27_2024.csv) | Pharmaceutical manufacturing (C27) | 326 |
| [CSMAR_C30_2024.csv](CSMAR_C30_2024.csv) | Nonmetallic mineral products manufacturing (C30) | 110 |

Files are UTF-8 CSVs with one firm-year observation per row. Read security codes as text to preserve leading zeros.

## Variables

| Original field | Meaning | Unit | Role |
|---|---|---|---|
| 证券代码 | Security code | Identifier | Firm identification |
| 证券简称 | Abbreviated company name | Text | Firm identification |
| 行业代码 | Industry code | C27 or C30 | Industry grouping |
| 行业名称 | Industry name | Text | Industry description |
| 年份 | Year | Calendar year | Observation year |
| 固定资产净额 | Net fixed assets | Million RMB | Capital input |
| 员工人数 | Number of employees | Persons | Labour input |
| 营业总收入 | Total operating revenue | Million RMB | Output |

## Preprocessing

The application uses net fixed assets and employee count as the two inputs and total operating revenue as the output. Retain observations with complete, strictly positive values for all three measures, and apply the natural-log transformation log(1 + value) to each measure. The CSVs retain the values before this transformation.

All 326 C27 records and 110 C30 records satisfy this complete-positive-value rule. The files have no missing cells or duplicate security-code/year keys. Industry estimation is performed separately for C27 and C30.

## Source

China Stock Market & Accounting Research database (CSMAR): https://www.csmar.com/channels/31.html . Please acknowledge CSMAR when using these data and cite the accompanying manuscript for the empirical design.
