
### Intro

- How you build LLMs like ChatGPT
- Cognitive implications of LLM tools


### How you build ChatGPT

Multiple stages arranged sequentially
- Pretraining
  - Step 1. Download and preprocess the Internet
    - [HuggingFace Dataset](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
      - All major LLM providers will have some internal dataset like the FineWeb Dataset
      - We want high quality of documents and large diversity of documents
      - **Disk Space you need:** FineWeb takes 44TB disk space because we filtered the Internet text quite aggresively - could potentially fit on a single hard drive
      - **Finding the raw data:** Start from the [CommonCrawl](https://commoncrawl.org/)
      - **Filtering and preprocessing steps**
        - URL Filtering
          - [Blocklists](https://dsi.ut-capitole.fr/blacklists/) where you don't wanna get data from (Malware, Spam, Marketing, Racist, Adult websites etc.)
        - Text Extraction
          - Basic Filtering - adequately filtering the good content from HTML pages.
        - Language Filtering
          - One example is only keep webpages with >= 65% of English text. But this filtering can be applied to other languages as well.
        - Gopher Filtering
        - Minhash Dedup
        - C4 Filters
        - Custom Filters
        - PII Removal
      - Dataset Preview
        - <img width="1016" alt="image" src="https://github.com/user-attachments/assets/86c5196b-d4ed-47ad-97d3-a2f883bef38b" />

- 
