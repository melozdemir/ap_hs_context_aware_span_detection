# ap_hs_context_aware_span_detection

This repository contains code and dataset for the paper:

**Context-aware span detection, binary classification, and election-period analysis**

> Annotation guidelines, models, and an analysis of the 15-day windows before the 2012 and 2016 U.S. elections

We provided: (1) annotation guidelines and a manually annotated Reddit dataset; (2) BERT-based classifiers for AP and HS trained with context/chunking; (3) scripts to extract election windows from the Reddit Politosphere dataset and run large-scale inference; and (4) plotting code to reproduce figures and tables in the paper.

---

## Repo layout

```
.
├── preprocessing/
│   ├── politosphere_to_election15day.py   # build 2012/2016 windows (parent–reply)
│   ├── hs_filtering_polito_to_sample_annotation.py                        # HS-keyword filtering sampler (2017–2019)
│   
│
├── ap_detection/
│   ├── context_ap_classifier_binary_bert_chunked.py   # reply or reply+context binary AP
│   ├── ap_detection_spanleftright.py                  # [SPAN][LCTX][RCTX] (+parent) AP classifier
│   └── ap_inference.py                                # batched inference over JSONL
│
├── hs_detection/
│   ├── hate_s_claffier.py                # BERT fine-tuning on Mody et al. (2022)
│   └── hs_inference.py                   # batched inference over JSONL
│
├── analysis/
│   ├── ap_analysis.py               
│   ├── co_occurence_analysis.py              
│   └── hs_analysis.py                          
│
├── dataset/
│   ├── annotated_data.jsonl                        # Reddit annotation JSONL(s)
│   
│
├── requirements.txt
└── README.md
```

For running the notebooks we used kaggle platform to be able to use GPU for free. 
```
## Used Datasets

* **Reddit Politosphere** (Hofmann et al., ICWSM’22).

* **Curated Hate Speech Dataset** (Mody et al., 2022).

* **Annotated Reddit set** .

```
