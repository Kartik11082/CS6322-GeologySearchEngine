# Crawling Module Details

Below are the detailed answers to the specific questions regarding the web crawling module and its architecture.

### Q1. How did you gather the web pages and how did you pass the collection to the index creation?
I built an asynchronous web crawler using Python's `asyncio` library. The crawler starts with a predetermined list of "Seed URLs" and pushes them into a Frontier (a queue of URLs to visit). A pool of asynchronous workers pulls URLs from the Frontier, downloads the HTML content, and extracts both the text and any outgoing hyperlinks. 

To pass this massive amount of data cleanly to the student building the indexing module, we established a "batching" architecture. Instead of passing one giant file or a complex database, the crawler continuously dumps the downloaded page data and hyperlink references into compressed line-delimited JSON files (e.g., `pages_batch_0.jsonl.gz` and `edges_batch_0.jsonl.gz`) inside a `crawled_data/` directory. The index creation script simply reads these compressed batches line-by-line using a loader script. 

**Example**:
When my crawler finishes fetching a page, it saves an entry to the JSON file formatted like this, which the indexer can immediately consume:
```json
{"url": "https://www.geology.com", "title": "Geology.com", "text": "Geoscience news and information...", "depth": 0}
```

### Q2. Describe clearly how many web pages you were able to crawl and make use in the search engine of your project.
Based on the final processing run through the indexer logic, we successfully crawled, parsed, and utilized **101,956 unique web pages**. Additionally, from those pages, we managed to extract and map **6,399,304 hyperlinks** (edges) pointing between documents, which successfully seeded the database for our search engine index.

### Q3. Provide details on the web pages that were sources for your crawls.
To ensure the search engine would heavily favor high-quality geological information, I manually defined dozens of highly authoritative "Seed URLs" spanning different geological disciplines to act as the root starting points for the crawler. 

These seed URLs were grouped into distinct geological categories so the crawler wouldn't stray into irrelevant parts of the web:
- **Universities with Geology Departments**: e.g., `https://www.geo.utexas.edu/`
- **Research Institutions & Government Surveys**: e.g., `https://earthquake.usgs.gov/earthquakes/`, `https://www.bgs.ac.uk/geology-projects/`
- **Volcanology & Paleontology databases**: e.g., `https://volcanoes.usgs.gov/`, `https://paleobiodb.org/`
- **Mineralogy Registries**: e.g., `https://www.mindat.org/`

By branching out exclusively from these initial authoritative boundaries, the crawler gathered a highly dense geology-specific web graph.

### Q4. Discuss how you made sure you did not have duplication in your crawl.
To prevent the crawler from getting stuck in infinite loops (e.g., Page A links to Page B, which links back to Page A) or wasting bandwidth downloading the same data twice, I implemented a strict `DedupFilter` (Deduplication Filter). 

Because checking a list of millions of URLs is computationally expensive, I maintained a fast-lookup tracking system in memory. Before an extracted URL is ever allowed to enter the Frontier queue, it is passed through `dedup.seen(link)`. If the exact URL has already been processed or is already in the queue, it is immediately discarded. Furthermore, I saved the deduplication tracker state so that if the crawler crashed and restarted, it could load the known URLs and resume without blindly re-downloading duplicate pages!

### Q5. Elaborate on how you provided hyperlink information for the student that generated the index and relevance models.
Relevance models like PageRank and HITS heavily rely on understanding which pages point to other pages. 

To provide this, every time the crawler's parser opened a webpage, it extracted every single `<a>` (anchor) tag's `href` attribute. I then recorded these as "Edges", noting the `src_url` (where the link was found) and the `dst_url` (where the link points). I dumped these explicit relationships into separate JSONL files named `edges_batch_*.jsonl.gz`.

**Example**:
If the crawler visited the USGS homepage and found a link pointing to Wikipedia, it would produce the following data structure in the edges file:
```json
{"src_url": "https://www.usgs.gov", "dst_url": "https://en.wikipedia.org/wiki/Geology"}
```
The indexing student simply mapped these string URLs into numerical Document IDs from their end to mathematically generate Directed Adjacency Graphs for their PageRank and HITS algorithms.
