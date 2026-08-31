# United Nations Citation Explorer

A pipeline for collecting, organizing, and cross-referencing legal texts from the United Nations and related international bodies — resolutions, treaties, conventions, protocols, and other instruments — and exploring how they cite one another.

## Project Structure

```
United_Nations_citation_explorer/
├── resolutions/ 
├── treaties/
├── instruments/
├── unesco_instruments/ 
├── conv-prot-rec/
├── document_embeddings/
├── Qwen3/
├── requirements.txt
└── LICENSE.txt
```

## Modules

### [`resolutions/`](./resolutions)
UN General Assembly resolutions from 1946 to 2019, along with their citations. Collected by Mesquita and Pires.

### [`treaties/`](./treaties)
Treaties collected from Wikipedia, as well as treaties collected from the UN Treaty Collection
### [`instruments/`](./instruments)
Instruments gathered by the [Office of the UN High Commissioner for Human Rights (OHCHR)](https://www.ohchr.org/en/). 
Many overlap with `resolutions/`, but this folder also covers other legal instruments beyond resolutions.

### [`unesco_instruments/`](./unesco_instruments)
Legal instruments issued by the United Nations Educational, Scientific and Cultural Organization (UNESCO) 

### [`conv-prot-rec/`](./conv-prot-rec)
International Labour Organization (ILO) conventions, protocols, and recommendations.

### [`document_embeddings/`](./document_embeddings)
Embeds documents and builds a FAISS index so they can be quickly matched against a natural-language topic description (e.g. "human rights", "gender equality") and retrieved.

### [`Qwen3/`](./Qwen3)
A minimal training script for a Qwen3 model. Kept intentionally simple — the actual model training runs on a server and a private repository; this folder just holds the essential script.

## License
This project is licensed under the terms in [`LICENSE.txt`](./LICENSE.txt) (MIT).
