# dannunzio-pipeline

Referring-expression pairing for a parallel corpus of Gabriele D'Annunzio's *Il trionfo della morte* (1894) and eight Japanese translations.

---

## Description of the contents

### Annotated text

| file | contents |
|---|---|
| `1894-dannunzio-01.xml` | Italian source, Book One (chs. I–VII), fully annotated |
| `1913-ishikawa-01.xml` | Ishikawa Gian's 1913 translation, same range, fully annotated |
| `1894_1913-alignments-01.xml` | stand-off sentence alignment between the two |

### Schema

| file | contents |
|---|---|
| `dannunzio-coref.odd.xml` | the customization as authored |
| `dannunzio-coref.rng` | RELAX NG schema generated from the ODD |
| `dannunzio-coref.isosch` | ISO Schematron constraints generated from the ODD |
| `dannunzio-coref.html` | documentation of the customization |

The customization adds two attributes to `<rs>`, `@pos` and `@case`, which are not defined in TEI P5. The corpus is therefore a *TEI Extension*. The Schematron additionally enforces that every `@ref` resolves to a `<person>` or `<personGrp>` in the header, that `@pos="NULL"` occurs only on subjects, and that alignment pointers resolve to existing sentence identifiers.

### Derived data

`data/*-mentions.csv` — one row per correspondence for each of the eight translations: a pair of mentions, or a single unpaired mention.

For the other seven translations, the columns holding running text (`source_sents`, `target_sents`, `source_text`, `target_text`) have been removed. Everything needed to recompute the published tables and figure is retained.

### Code

| file | purpose |
|---|---|
| `pairing.py` | the pairing algorithm |
| `analysis.ipynb` | notebook that computes the conditional probabilities for how determiners, non-reflexive pronouns and zero subjects were translated, using the mentions' CSVs, and runs a clustering algorithm based on those values |

## Running it

```bash
pip install -r requirements.txt

python pairing.py \
    --source     1894-dannunzio-01.xml \
    --target     1913-ishikawa-01.xml \
    --alignment  1894_1913-alignments-01.xml \
    --out        data/1894-1913-mentions.csv
```

`pairing.py` prints two checks before writing. *Coverage* reports the proportion of annotated mentions that fall inside an aligned sentence block. Mentions in unaligned sentences are excluded from the analysis. 
*Conservation* verifies that every mention inside an aligned block appears exactly once in the output, as one half of a pair or as a leftover.

### Caution

`NULL` is in pandas' default missing-value sentinel list, so a plain `pd.read_csv` turns every zero-subject annotation into `NaN` and the next write saves it as an empty field. Always read these files with:

```python
pd.read_csv(path, keep_default_na=False, na_values=[])
```

`analysis.ipynb` does this by default.

### Reproducibility

`EMBEDDER_REVISION` in `pairing.py` pins the sentence-embedding model to a specific revision. Leaving it unpinned means a future update to the model changes the cost values, and with them the cost deciles reported in Tab. 3.

Exact package versions used for the published results are in `requirements.txt`.

---

## Validating the TEI

The files carry two `<?xml-model?>` instructions, one for RELAX NG and one for Schematron. Both schemas must be present alongside the XML for validation to work.

To confirm the Schematron is actually firing, change one `@ref` to a value not
present in `<listPerson>` and check that the validator reports it.

---

## Licence

Code (`pairing.py`, `analysis.ipynb`): MIT, see `LICENSE`.

Annotation, encoding, alignment and derived data: CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/).

The Italian source text is in the public domain. The underlying Japanese translations are the property of their respective rights holders and are not redistributed here.

---

## Citation

See `CITATION.cff`.
