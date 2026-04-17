from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    TARGET_PAGES: int = 100_000
    CONCURRENCY: int = 60
    MAX_DEPTH: int = 10
    REQUEST_TIMEOUT: float = 12.0
    CONNECT_TIMEOUT: float = 6.0
    MAX_RETRIES: int = 2
    RETRY_BACKOFF: float = 1.2
    DOMAIN_DELAY: float = 1.0
    MAX_PAGE_SIZE_MB: float = 3.0
    ALLOWED_CONTENT_TYPES: tuple[str, ...] = ("text/html", "application/xhtml+xml")

    # URLs must contain at least one of these to be crawled
    FOCUS_KEYWORDS: tuple[str, ...] = (
        "geolog",
        "geology",
        "earth-science",
        "earthscience",
        "mineral",
        "mineralogy",
        "volcano",
        "volcanic",
        "volcan",
        "seismic",
        "seismology",
        "earthquake",
        "tecton",
        "tectonic",
        "rock-",
        "rocks",
        "fossil",
        "fossils",
        "sediment",
        "sedimentary",
        "geomorph",
        "stratigraphy",
        "stratigraph",
        "petrology",
        "petrograph",
        "geophysic",
        "geochemist",
        "hydrogeol",
        "groundwater",
        "usgs",
        "bgs.ac.uk",
        "noaa.gov",
        "iris.edu",
        "serc.carleton",
        "geonet",
        "geoscience",
        "geo-science",
        "earthobservatory",
        "mindat",
        "litholog",
        "geohazard",
        "geothermal",
        "paleontol",
        "paleozoic",
        "mesozoic",
        "cenozoic",
        "igneous",
        "metamorphic",
        "magma",
        "lava",
        "fault",
        "bedrock",
    )

    # URLs containing ANY of these are immediately discarded - no exceptions
    JUNK_URL_PATTERNS: tuple[str, ...] = (
        "/contact",
        "/about-us",
        "/about_us",
        "contact-us",
        "contactus",
        "/team",
        "/staff",
        "/careers",
        "/jobs",
        "/employment",
        "/privacy",
        "/terms",
        "/legal",
        "/disclaimer",
        "/cookie",
        "/advertis",
        "/sponsor",
        "/partner",
        "/login",
        "/signin",
        "/signup",
        "/register",
        "/account",
        "/cart",
        "/checkout",
        "/shop",
        "/store",
        "/product",
        "/social",
        "/twitter",
        "/facebook",
        "/instagram",
        "/linkedin",
        "/youtube",
        "/tiktok",
        "/pinterest",
        "share=",
        "utm_",
        "ref=",
        "source=social",
        "/newsletter",
        "/subscribe",
        "/unsubscribe",
        "/sitemap",
        "/tag/",
        "/author/",
        "/wp-admin",
        "/cdn-cgi",
        "/static/",
        "/assets/",
        "javascript:",
        "mailto:",
        "tel:",
    )

    # Page text must contain at least one of these or it is discarded after fetch
    CONTENT_KEYWORDS: tuple[str, ...] = (
        "geology",
        "geological",
        "geologic",
        "mineral",
        "rock",
        "sediment",
        "fossil",
        "earthquake",
        "seismic",
        "volcano",
        "volcanic",
        "tectonic",
        "stratigraphy",
        "lithology",
        "igneous",
        "metamorphic",
        "sedimentary",
        "geophysics",
        "geochemistry",
        "paleontology",
        "earth science",
        "earth sciences",
        "fault",
        "magma",
        "lava",
        "bedrock",
        "groundwater",
    )

    OUTPUT_DIR: str = "output"
    PAGES_FILE: str = "pages.jsonl"
    EDGES_FILE: str = "web_graph.csv"
    STATS_FILE: str = "crawl_stats.json"
    BLOOM_CAPACITY: int = 5_000_000
    BLOOM_ERROR_RATE: float = 0.001
