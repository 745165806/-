# Third-party sources and licensing boundary

The project contains original integration/baseline code and a small domain schema. Version 0.2 also vendors the two expert implementations listed below. It does not bundle pretrained weights or speech corpora. The MIT license for original project code does not relicense third-party materials.

## Vendored utterance experts

- `src/ontoforensics/utterance/vendor/aasist.py`: NAVER AASIST (`https://github.com/clovaai/aasist`, upstream `models/AASIST.py`), MIT, copyright NAVER Corp. Original license retained as `AASIST_LICENSE`. The wrapper uses this architecture with the project's own training labels and checkpoints.
- `src/ontoforensics/utterance/vendor/ssl.py`: Hemlata Tak et al., SSL Anti-spoofing (`https://github.com/TakHemlata/SSL_Anti-spoofing`, upstream `model.py`), MIT, copyright 2022 Hemlata. Original license retained as `SSL_LICENSE`. The checkpoint path is changed to `ONTO_XLSR_CHECKPOINT`; the upstream fairseq dependency and XLS-R weights must be provided separately.

These are research integrations, not reproductions of the authors' published training recipes or results. The supplied trainer does not implement upstream RawBoost or an identical learning-rate schedule. Source hashes and local upstream revisions are recorded in `examples/vendor_provenance.json`.

Primary references checked for this implementation (2026-11-06):

| Component | Primary source | Use in this project |
|---|---|---|
| PartialEdit v1.1 | https://zenodo.org/records/18829689 | Annotation and speaker protocol format; local importer |
| PartialEdit paper | https://www.isca-archive.org/interspeech_2025/zhang25g_interspeech.pdf | Task, editing/codec distinction and evaluation motivation |
| VCTK 0.92 metadata | https://datashare.ed.ac.uk/handle/10283/3443 | Source-corpus reference; retrieve audio under its own terms |
| CFPRF | https://github.com/ItzJuny/CFPRF | External FDN interface; upstream repository labels code MIT |
| PartialSpoof | https://github.com/nii-yamagishilab/PartialSpoof | Score interface and future expert integration; upstream repository labels code BSD-3-Clause |
| PartialSpoof data | https://zenodo.org/records/5766198 | Future data adapter; not implemented in this version |
| PROV-O | https://www.w3.org/TR/prov-o/ | Provenance classes and properties |
| SOSA/SSN | https://www.w3.org/TR/vocab-ssn/ | Result/observation semantic distinction |
| SHACL | https://www.w3.org/TR/shacl/ | RDF shape validation |
| scikit-learn calibration | https://scikit-learn.org/stable/modules/calibration.html | Background for disjoint calibration and probabilistic evaluation |

Consult each upstream source for current code, checkpoint, dataset and component licenses. Upstream version changes can invalidate an adapter; the CFPRF exporter records relevant file hashes and checks native output shape, but an actual checkpoint smoke test is still required.
