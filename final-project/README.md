# From Lab to Lecture:

## Analyzing the Connection Between Professors’ Research and Course Content​ 👩‍🏫

**Authors**:

-   Erik Wold Riise, s194633​
-   Lukas Rasocha, s233498​
-   Zou Yong Nan Klaassen, s230351

### Project Overview ✍️

This project investigates the alignment between professors’ research areas and the courses they teach through the angle of network analysis and natural language processing (NLP).
We plan to construct a bipartite graph of professors and courses, and analyze the structural and thematic patterns in teaching and research connections.

The central research question steering the project is:
_"How well do professors’ research areas align with the content and objectives of the courses they teach, and how does this alignment vary across disciplines?"_

To complement this, we also examine:
_"Does the alignment between professors’ research and the courses they teach influence student satisfaction and performance (grades)?"_

Using NLP techniques, we analyze course descriptions and research topics to measure alignment, and we relate these findings to course evaluations and grades. Additionally, network analysis methods, such as community detection and centrality measures, will be applied to uncover interdisciplinary trends and the influence of professors within the academic network.

By this we hope to shed light on how expertise and teaching intersect, and how does that impact educational outcomes in a broader sense.

### Setup 🛠️

1. Clone the repository
2. Ensure you are using Python 3.9
3. Install the required packages by running:

```bash
pip install -r requirements.txt
```

### How to use our scraper

You need to have `python=3.9`

```python
scraper = DTUOrbitScraper()
profile_info = scraper.get_profile_info("Sune Lehmann")
```

### The notebook

All the analysis and code to reproduce the results in our paper can be found in the `main.ipynb` notebook.
