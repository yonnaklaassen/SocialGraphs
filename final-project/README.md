# Final Project: Are professors happy?

## Introduction

We will use the data from DTU courses and the data from the research of professors to create a network of courses and professors. We will then analyze the network to see if professors are teaching courses they are passionate about. We will also analyze the network to see if professors are teaching courses that are related to their research.

## How to use our scraper

You need to have `python=3.9` (because of scholia).

1. Install scholia:

```bash
python3 -m pip install git+https://github.com/WDscholia/scholia
```

2. Install beatifulsoup4:

```bash
pip install beautifulsoup4
```

3. Use scraper

```python
scraper = DTUOrbitScraper()
profile_info = scraper.get_profile_info("Ole Winther")
```
