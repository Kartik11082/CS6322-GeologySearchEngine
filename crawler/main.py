import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import ssl
import time
from pathlib import Path

if not sys.platform.startswith("win"):
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

from crawler.config import Config
from crawler.dedup import DedupFilter
from crawler.fetcher import Fetcher
from crawler.frontier import Frontier
from crawler.parser import Parser
from crawler.storage import Storage

# SEED_URLS = (
#     "https://www.usgs.gov/science/science-explorer/Natural_Hazards",
#     "https://www.usgs.gov/science/science-explorer/Earth_Processes",
#     "https://earthquake.usgs.gov/earthquakes/",
#     "https://volcanoes.usgs.gov/",
#     "https://www.geology.com/",
#     "https://www.geosociety.org/",
#     "https://earthobservatory.nasa.gov/",
#     "https://www.mindat.org/",
#     "https://www.geologypage.com/",
#     "https://www.bgs.ac.uk/geology-projects/",
#     "https://serc.carleton.edu/geoscience.html",
#     "https://www.ngdc.noaa.gov/",
#     "https://volcano.oregonstate.edu/",
#     "https://www.volcanodiscovery.com/",
#     "https://www.geolsoc.org.uk/",
#     "https://www.americangeosciences.org/",
#     "https://www.earthmagazine.org/",
#     "https://geonet.org.nz/",
#     "https://www.mineralogicalassociation.ca/",
#     "https://www.geotimes.org/",
# )

SEED_URLS = (
    # --- already crawled, keep for continuity ---
    "https://www.usgs.gov/science/science-explorer/Natural_Hazards",
    "https://earthquake.usgs.gov/earthquakes/",
    "https://volcanoes.usgs.gov/",
    "https://www.geology.com/",
    "https://www.mindat.org/",
    "https://earthobservatory.nasa.gov/",
    "https://www.bgs.ac.uk/geology-projects/",
    # --- universities with geology departments ---
    "https://www.geo.utexas.edu/",
    "https://eps.berkeley.edu/",
    "https://earth.stanford.edu/",
    "https://geo.arizona.edu/research/our-research/geology",
    "https://eps.harvard.edu/",
    "https://www.earth.columbia.edu/",
    "https://geology.utah.edu/",
    "https://www.geology.wisc.edu/",
    "https://www.soest.hawaii.edu/GG/",
    "https://geology.indiana.edu/",
    "https://www.geological.lsu.edu/",
    "https://www.geo.umass.edu/",
    "https://ees.arizona.edu/",
    "https://www.geo.cornell.edu/",
    "https://earthsciences.dartmouth.edu/",
    "https://geology.colostate.edu/",
    "https://www.umt.edu/geosciences/",
    "https://geology.ku.edu/",
    "https://www.nmt.edu/academics/ees/",
    "https://www.geology.ohio-state.edu/",
    "https://geology.rutgers.edu/",
    # --- research institutions ---
    "https://www.ldeo.columbia.edu/",
    "https://www.whoi.edu/",
    "https://www.mbari.org/",
    "https://www.unavco.org/",
    "https://www.iris.edu/hq/",
    "https://www.seis.nagoya-u.ac.jp/",
    "https://www.gfz-potsdam.de/en/",
    "https://www.nrcan.gc.ca/earth-sciences/",
    "https://www.ga.gov.au/",
    "https://www.gns.cri.nz/",
    "https://www.niwa.co.nz/",
    "https://www.igs.ac.cn/english/",
    # --- government geological surveys ---
    "https://www.geologicalsurvey.ie/",
    "https://www.ngu.no/en/",
    "https://www.gtk.fi/en/",
    "https://www.sgu.se/en/",
    "https://www.geus.dk/en/",
    "https://www.gsq.qld.gov.au/",
    "https://www.mgs.md.gov/geology/",
    "https://www.geology.illinois.edu/isgs/",
    "https://geology.com/state-map/",
    "https://www.kgs.ku.edu/",
    "https://www.isgs.illinois.edu/",
    "https://www.conservation.ca.gov/cgs",
    "https://www.mgs.state.ms.us/",
    # --- mineralogy and petrology ---
    "https://www.mindat.org/mineral.php",
    "https://rruff.info/",
    "https://www.handbookofmineralogy.org/",
    "https://www.minerals.net/",
    "https://geology.com/minerals/",
    "https://www.gemdat.org/",
    "https://www.minsocam.org/",
    "https://www.mineralogicalsociety.org/",
    # --- volcanology ---
    "https://www.volcanolive.com/",
    "https://www.volcano.si.edu/",
    "https://www.volcanodiscovery.com/volcanoes.html",
    "https://www.geo.mtu.edu/volcanoes/",
    "https://volcano.oregonstate.edu/vwdocs/volc_images/",
    "https://www.avo.alaska.edu/",
    "https://hvo.wr.usgs.gov/",
    "https://www.globalvolcanomodel.org/",
    # --- seismology and geophysics ---
    "https://www.seismosoc.org/",
    "https://www.isc.ac.uk/",
    "https://earthquake.usgs.gov/data/",
    "https://www.emsc-csem.org/",
    "https://ds.iris.edu/ds/nodes/dmc/",
    "https://www.agu.org/",
    "https://www.earthscope.org/",
    # --- paleontology and stratigraphy ---
    "https://paleobiodb.org/",
    "https://www.fossilworks.org/",
    "https://www.nhm.ac.uk/our-science/departments-and-staff/palaeontology.html",
    "https://www.amnh.org/our-research/paleontology",
    "https://strata.geology.wisc.edu/",
    "https://www.stratigraphy.org/",
    "https://timescalefoundation.org/",
    "https://www.geologic-time-scale.org/",
    # --- journals and publications ---
    "https://www.geoscienceworld.org/",
    "https://pubs.geoscienceworld.org/",
    "https://www.geologysociety.org/",
    "https://www.nature.com/ngeo/",
    "https://agupubs.onlinelibrary.wiley.com/",
    "https://www.sciencedirect.com/journal/earth-science-reviews",
    "https://www.tandfonline.com/toc/tgeo20/current",
    "https://gji.oxfordjournals.org/",
    # --- education and outreach ---
    "https://serc.carleton.edu/",
    "https://www.earthsciweek.org/",
    "https://www.scienceclarified.com/geosphere/",
    "https://www.windows2universe.org/earth/",
    "https://www.geolsoc.org.uk/Education-and-Careers/",
    "https://www.americangeosciences.org/education",
    "https://geology.com/articles/",
    "https://www.geologypage.com/category/rocks",
    "https://www.geologypage.com/category/minerals",
    "https://www.geologypage.com/category/volcanoes",
    # --- climate and earth systems ---
    "https://climate.nasa.gov/",
    "https://www.noaa.gov/education/resource-collections/climate",
    "https://nsidc.org/",
    "https://www.ncdc.noaa.gov/",
    "https://www.gfdl.noaa.gov/",
    "https://www.cgd.ucar.edu/",
    # --- geomorphology and soils ---
    "https://www.nrcs.usda.gov/wps/portal/nrcs/site/soils/home/",
    "https://www.geomorphology.org.uk/",
    "https://www.iag-iag.org/",
    "https://www.soilsciencesociety.org/",
    # --- hydrology and groundwater ---
    "https://water.usgs.gov/",
    "https://www.ngwa.org/",
    "https://www.awra.org/",
    "https://www.hydroworld.com/",
    "https://www.internationalgroundwaterresourcescenter.org/",
    # --- geohazards ---
    "https://www.bgs.ac.uk/geology-projects/geohazards/",
    "https://www.preventionweb.net/",
    "https://landslides.usgs.gov/",
    "https://www.tsunamiready.noaa.gov/",
    "https://www.itic.ioc-unesco.org/",
)


async def run_crawler() -> None:
    if sys.platform.startswith("win"):
        loop = asyncio.get_running_loop()

        def _windows_exception_handler(
            loop: asyncio.AbstractEventLoop, context: dict
        ) -> None:
            exc = context.get("exception")
            message = str(context.get("message", ""))

            # Known harmless Windows Proactor shutdown noise from remote resets.
            if isinstance(exc, ConnectionResetError):
                return
            if isinstance(exc, ssl.SSLError) and "_call_connection_lost" in message:
                return
            if "_ProactorBasePipeTransport._call_connection_lost" in message:
                return

            loop.default_exception_handler(context)

        loop.set_exception_handler(_windows_exception_handler)

    cfg = Config()
    frontier = Frontier(cfg)
    fetcher = Fetcher(cfg)
    parser = Parser(cfg)
    dedup = DedupFilter(cfg)
    storage = Storage(cfg)
    dedup.load_from_existing(str(Path(cfg.OUTPUT_DIR) / cfg.PAGES_FILE))

    # Always enqueue seeds as bootstrap pages, even if already seen in a
    # previous run. This allows discovery of newly linked pages without
    # needing persisted frontier checkpoints.
    queued_seeds: set[str] = set()
    for seed in SEED_URLS:
        if seed in queued_seeds:
            continue
        queued_seeds.add(seed)
        if not dedup.seen(seed):
            dedup.add(seed)
        await frontier.push(seed, depth=0)

    semaphore = asyncio.Semaphore(cfg.CONCURRENCY)
    active_tasks: set[asyncio.Task] = set()

    crawled = 0
    next_log_at = 500
    started_at = time.monotonic()

    async def worker(url: str, depth: int) -> None:
        nonlocal crawled, next_log_at

        async with semaphore:
            result = await fetcher.fetch(url)
        if result is None:
            return

        try:
            links, text, title = parser.parse(result["html"], url)
        except Exception:
            return
        if text == "":
            return

        await storage.save_page(
            {
                "url": url,
                "title": title,
                "text": text,
                "content_type": result["content_type"],
                "crawled_at": result["crawled_at"],
                "status": result["status"],
                "depth": depth,
            }
        )
        await storage.save_edges(url, links)

        crawled += 1
        if crawled >= next_log_at:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            rate = crawled / elapsed
            remaining = max(cfg.TARGET_PAGES - crawled, 0)
            eta_min = (remaining / rate) / 60 if rate > 0 else 0.0
            print(
                f"✓ {crawled:,} pages | {rate:.1f} p/s | ETA {eta_min:.0f} min | Queue: {frontier.size():,}"
            )
            next_log_at += 500

        if depth >= cfg.MAX_DEPTH:
            return

        for link in links:
            if dedup.seen(link):
                continue
            dedup.add(link)
            await frontier.push(link, depth + 1)

    try:
        while True:
            if crawled >= cfg.TARGET_PAGES:
                for task in active_tasks:
                    task.cancel()
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                active_tasks.clear()
                break

            while len(active_tasks) < cfg.CONCURRENCY and crawled < cfg.TARGET_PAGES:
                item = await frontier.pop()
                if item is None:
                    break
                url, depth = item
                task = asyncio.create_task(worker(url, depth))
                active_tasks.add(task)

            if not active_tasks:
                if frontier.size() == 0:
                    break
                await asyncio.sleep(0.05)
                continue

            done, _ = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                active_tasks.discard(task)
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    print(f"Task error: {exc}")
    finally:
        await fetcher.close()
        await storage.close()

    elapsed_sec = max(time.monotonic() - started_at, 1e-9)
    elapsed_min = elapsed_sec / 60.0
    avg_rate = crawled / elapsed_sec
    print("Crawl complete")
    print(f"Total pages saved: {crawled:,}")
    print(f"Total unique URLs seen: {dedup.count():,}")
    print(f"Elapsed time: {elapsed_min:.2f} minutes")
    print(f"Average rate: {avg_rate:.2f} p/s")


if __name__ == "__main__":
    asyncio.run(run_crawler())
