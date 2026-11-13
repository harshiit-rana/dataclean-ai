# The real-error benchmark (#127)

The [cleaning benchmark](cleaning.md) injects the problems the profiler was built to find. These datasets carry
errors people actually made. Each comes as a dirty file and a clean version made by hand, from the error-detection
research of Mahdavi et al. (Raha, SIGMOD 2019; Baran, VLDB 2020), published at
[github.com/BigDaMa/raha](https://github.com/BigDaMa/raha) under Apache-2.0.

| Dataset | Rows | Kinds of error |
|---|---:|---|
| hospital | 1,000 | typos (letters swapped for "x") in almost every column |
| flights | 2,376 | missing and conflicting departure and arrival times, from different sources |
| beers | 2,410 | units glued to numbers ("12.0 oz"), blanks, city and state in one field |
| rayyan | 1,000 | blanks, wrong dates and volumes in bibliographic records |
| movies_1 | 7,390 | thousands separators ("379,998"), inconsistent formats, swapped values |

An erroneous cell is one whose dirty value differs from its clean one, compared the way Raha compares them: as text,
with HTML entities decoded and runs of whitespace collapsed.

## How it is scored

**Detection, per cell**, as in the papers. The profiler reports problems per column, so each finding is turned into
the cells it points at:
- the blanks, for missing values;
- the values that do not parse, for numbers or dates stored as text;
- the values outside its range, for outliers;
- the rarer spelling, for inconsistent capitals;
- the padded values, for stray spaces;
- the rarer shape, for mixed date formats.

Duplicates and high cardinality point at no cell. Precision, recall and F1 are measured against the erroneous cells,
plus how many of the columns with errors get any finding.

**Repair, per cell.** The fix each finding implies (the cleaning benchmark's reference fixes, without dropping
duplicate rows, which would misalign the files) is applied to the whole dirty file. Each changed cell is compared
with the clean one: dates by the moment they name, numbers by value. Three outcomes are counted:
- **right**: an erroneous cell now matches;
- **overwritten**: a correct written value no longer matches;
- **invented**: a blank that was really blank is now filled. That is what filling does, so it is counted apart.

The profiler was never meant to catch typos or values that are wrong but well formed, and most of these errors are
of that kind, so low recall is expected. What the benchmark shows is how much of a real file's damage the app can
see, and whether what it fixes, it fixes without breaking the rest.

```bash
cd backend
python scripts/real_errors_benchmark.py          # about 20 seconds once the files are downloaded
```

## Results (4 November 2026)

| Dataset | Erroneous cells | Detection P | R | F1 | Columns with errors flagged | Repair P | R | Correct values overwritten | Real blanks filled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hospital | 509 | 1.000 | 0.181 | 0.306 | 3 of 17 | 0.000 | 0.000 | 0 | 0 |
| flights | 4,920 | 1.000 | 0.470 | 0.640 | 4 of 4 | 0.019 | 0.009 | 0 | 0 |
| beers | 4,362 | 0.947 | 0.295 | 0.449 | 4 of 5 | 0.013 | 0.004 | 5 | 67 |
| rayyan | 948 | 0.081 | 0.169 | 0.110 | 7 of 8 | 0.000 | 0.000 | 155 | 1,607 |
| movies_1 | 7,675 | 0.147 | 0.060 | 0.086 | 8 of 11 | 0.001 | 0.002 | 1 | 2,680 |

Cells flagged and right, by kind of finding (all datasets):

- date_formats: 132 flagged, 85 erroneous (64%)
- inconsistent_casing: 205 flagged, 157 erroneous (77%)
- missing_values: 7,980 flagged, 3,626 erroneous (45%)
- outliers: 112 flagged, 0 erroneous (0%)
- stray_whitespace: 3 flagged, 0 erroneous (0%)
- type_mismatch: 447 flagged, 447 erroneous (100%)

**What it says.**

- **What the profiler flags is mostly right where its kind of problem exists.** Every cell it called a number
  stored as text was an error (447 of 447), three in four of its capitals findings and two in three of its date
  findings were. On hospital and flights every flagged cell was an error.
- **It sees a small share of the damage**: 6% to 47% of the erroneous cells. Most of these errors are typos, values
  from the wrong source and well-formed wrong values, which a profiler cannot see without knowing the right answer.
  This is the expected result, not a defect; Raha and Baran reach their scores by learning from labelled cells.
- **Blanks are where precision falls.** On rayyan and movies_1 most blanks are blank in the clean file too, so they
  count as flagged but not erroneous, and filling them "invents" a value: 1,607 and 2,680 cells. The app proposes a
  fill and the user approves it, so this is what approving a fill means on such a file, not a silent change.
- **Outliers flagged were never errors here** (0 of 112). The extreme values in these files are real.
- **Fixing breaks little.** Before #129 and #130 the fixes overwrote 7,937 correct values (4,583 times erased on
  flights, 2,271 re-cased movie names, 821 on rayyan, 262 on beers). After them: 161. Of rayyan's 155, 109 are real
  volume and issue numbers capped as outliers (the same 0-of-112 problem: the extremes are real), and 46 are
  language names, journal titles and one abbreviation merged into another spelling the clean file keeps apart.
- **Repair barely moves the score** (repair recall under 1%). Only a fix whose target value is knowable from the file
  itself (a parsed number, a unified spelling, one date format) can be right, and those are a small share of what
  people get wrong.
