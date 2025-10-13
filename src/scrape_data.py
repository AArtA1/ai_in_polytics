import hashlib
import json
import logging
import os
import queue
import re
import time
import urllib.robotparser as robotparser
from dataclasses import asdict, dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from slugify import slugify
from tqdm import tqdm

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except Exception:
    TRAFILATURA_AVAILABLE = False


BASE_URL = "https://spravedlivo.ru/main"
DOMAIN = "spravedlivo.ru"
START_URL = BASE_URL

# Ключевые темы/якоря, по которым фильтруем и приоритезируем страницы
TOPIC_PATTERNS = [
    r"предвыборн",       # предвыборная программа/платформа
    r"программ[аы]|платформ",  # программа партии, платформа
    r"истори",           # история партии
    r"устав",            # устав
    r"идеолог",          # идеология
    r"цель|задач",       # цели/задачи
    r"социальн|экономик|внешн|внутренн",  # разделы программ
]

# Папки для сохранения
OUT_DIR = "data_spravedlivo"
HTML_DIR = os.path.join(OUT_DIR, "html")
TEXT_DIR = os.path.join(OUT_DIR, "text")
META_PATH = os.path.join(OUT_DIR, "metadata.jsonl")
LOG_PATH = os.path.join(OUT_DIR, "scrape.log")

# Сетевые настройки
HEADERS = {
    "User-Agent": "ResearchBot/1.0 (+https://example.org; contact: research@example.org)"
}
REQUEST_TIMEOUT = 15
RATE_LIMIT_SEC = 1.0    # пауза между запросами
MAX_PAGES = 1500        # предел обхода
MAX_DEPTH = 4           # чтобы не «провалиться» слишком глубоко
CONNECT_RETRIES = 2


@dataclass
class PageRecord:
    url: str
    status: int
    referer: str | None
    title: str | None
    path_html: str | None
    path_text: str | None
    detected_topics: list[str]
    depth: int
    fetched_at: float


def ensure_dirs():
    os.makedirs(HTML_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def setup_logging():
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)


def allowed_by_robots(url: str) -> bool:
    # проверяем robots.txt
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception as e:
        logging.warning(f"robots.txt check failed: {e}")
        # при ошибке — консервативно разрешим (или наоборот запретим — на ваш выбор)
        return True


def normalize_url(base: str, href: str) -> str | None:
    if not href:
        return None
    href = href.strip()
    if href.startswith("mailto:") or href.startswith("tel:"):
        return None
    full = urljoin(base, href)
    p = urlparse(full)
    if p.netloc and p.netloc != DOMAIN:
        return None  # внешние домены не обходим
    # убираем якоря и лишние параметры трекинга
    full = full.split("#")[0]
    return full


def hash_path(url: str, prefix: str, ext: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    slug = slugify(urlparse(url).path or "index")
    fname = f"{slug[:80]}_{h}.{ext}"
    return os.path.join(prefix, fname)


def detect_topics(text: str) -> list[str]:
    text_low = text.lower()
    found = []
    for pat in TOPIC_PATTERNS:
        if re.search(pat, text_low):
            found.append(pat)
    return found


def extract_title(soup: BeautifulSoup) -> str | None:
    # пытаемся вытащить <title> или h1
    t = soup.find("title")
    if t and t.text.strip():
        return t.text.strip()
    h1 = soup.find(["h1", "h2"])
    if h1 and h1.text.strip():
        return h1.text.strip()
    return None


def extract_text_basic(soup: BeautifulSoup) -> str:
    # базовая очистка: убираем скрипты/стили/навигацию
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    # часто полезны article/section/main
    main = soup.find(["article", "main", "section"]) or soup.body
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    # нормализация пустых строк
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_text_with_trafilatura(url: str, html: str) -> str | None:
    if not TRAFILATURA_AVAILABLE:
        return None
    try:
        downloaded = trafilatura.extract(html, include_links=False, include_tables=False, url=url)
        return downloaded
    except Exception:
        return None


def fetch(url: str) -> requests.Response | None:
    s = requests.Session()
    s.headers.update(HEADERS)
    for i in range(CONNECT_RETRIES + 1):
        try:
            resp = s.get(url, timeout=REQUEST_TIMEOUT)
            return resp
        except requests.RequestException as e:
            logging.warning(f"fetch error ({i+1}/{CONNECT_RETRIES}) {url}: {e}")
            time.sleep(1.5)
    return None


def crawl():
    ensure_dirs()
    setup_logging()
    logging.info("Start crawling")

    if not allowed_by_robots(START_URL):
        logging.error("Blocked by robots.txt for start URL")
        return

    seen: set[str] = set()
    q = queue.Queue()
    q.put((START_URL, None, 0))  # url, referer, depth
    pages_saved = 0

    with open(META_PATH, "w", encoding="utf-8") as meta_out:
        pbar = tqdm(total=MAX_PAGES, desc="Crawling")
        while not q.empty() and pages_saved < MAX_PAGES:
            url, referer, depth = q.get()

            if url in seen:
                continue
            seen.add(url)

            if depth > MAX_DEPTH:
                continue

            if not allowed_by_robots(url):
                logging.info(f"robots disallow: {url}")
                continue

            time.sleep(RATE_LIMIT_SEC)

            resp = fetch(url)
            if resp is None:
                logging.info(f"skip (no response): {url}")
                continue

            status = resp.status_code
            if status != 200 or not resp.content:
                logging.info(f"skip (status {status}): {url}")
                continue

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                logging.info(f"skip (non-html): {url}")
                continue

            html = resp.text
            soup = BeautifulSoup(html, "lxml")
            title = extract_title(soup)
            raw_text = None

            # Пробуем trafilatura для лучшего извлечения, иначе basic
            if TRAFILATURA_AVAILABLE:
                raw_text = extract_text_with_trafilatura(url, html)
            if not raw_text:
                raw_text = extract_text_basic(soup)

            topics = detect_topics((title or "") + "\n" + raw_text)

            # Сохраняем файлы
            html_path = hash_path(url, HTML_DIR, "html")
            txt_path = hash_path(url, TEXT_DIR, "txt")

            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

            rec = PageRecord(
                url=url,
                status=status,
                referer=referer,
                title=title,
                path_html=os.path.relpath(html_path, OUT_DIR),
                path_text=os.path.relpath(txt_path, OUT_DIR),
                detected_topics=topics,
                depth=depth,
                fetched_at=time.time(),
            )
            meta_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            pages_saved += 1
            pbar.update(1)

            # План расширения ссылок: только внутри домена
            for a in soup.find_all("a", href=True):
                nxt = normalize_url(url, a["href"])
                if not nxt:
                    continue
                # лёгкая фильтрация мусора
                if any(seg in nxt for seg in ("/search?", "/tag/", "/?PAGEN_", "/page/", "/feed")):
                    # при необходимости ослабьте фильтр
                    pass
                if nxt not in seen:
                    q.put((nxt, url, depth + 1))

        pbar.close()
    logging.info(f"Done. Saved pages: {pages_saved}")


if __name__ == "__main__":
    crawl()
