import requests
from bs4 import BeautifulSoup
import time
import json
import random
import logging
import re
import argparse
from pathlib import Path
from urllib.parse import quote
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ──────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ──────────────────────────────────────────────

BASE_URL = "https://cookpad.com/vn"
OUTPUT_DIR = Path("cookpad_data")

# Thời gian nghỉ giữa các request (giây)
DELAY_MIN = 0.3
DELAY_MAX = 0.8

# Số recipe tối đa cần cào (0 = không giới hạn)
MAX_RECIPES = 0

# Thông tin liên hệ của researcher
RESEARCHER_INFO = "Vietnamese Recipe Dataset for Academic Research"

HEADERS = {
    "User-Agent": (
        f"AcademicResearchBot/1.0 "
        f"({RESEARCHER_INFO})"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Từ khóa tìm kiếm — bao quát ẩm thực Việt Nam
VIETNAMESE_KEYWORDS = [
    # # ─────────── MÓN NƯỚC ───────────
    # "phở", "phở bò", "phở gà", "phở xào", "phở cuốn",
    # "bún", "bún bò", "bún chả", "bún riêu", "bún cá", "bún ốc",
    # "bún thịt nướng", "bún đậu mắm tôm", "bún thang",
    # "hủ tiếu", "hủ tiếu nam vang", "hủ tiếu khô",
    # "mì", "mì xào", "mì nước", "mì quảng",
    # "miến", "miến gà", "miến xào",
    # "bánh canh", "cháo",

    # # ─────────── CƠM ───────────
    # "cơm", "cơm tấm", "cơm chiên", "cơm rang",
    # "cơm gà", "cơm sườn", "cơm niêu",

    # # ─────────── CHẾ BIẾN ───────────
    # "món luộc", "luộc", "rau luộc", "thịt luộc", "gà luộc",
    # "món hấp", "hấp", "cá hấp", "tôm hấp", "nghêu hấp",
    # "món chiên", "chiên", "chiên giòn", "chiên xù",
    # "món xào", "xào", "xào tỏi", "xào sa tế",
    # "món kho", "kho", "kho tiêu", "kho tàu",
    # "món nướng", "nướng", "nướng than", "nướng giấy bạc",
    # "món rang", "rang", "rang muối", "rang me",
    # "món rim", "rim", "rim mặn ngọt",
    # "áp chảo", "om", "hầm",

    # # ─────────── THEO PROTEIN ───────────
    # "gà", "gà chiên", "gà nướng", "gà luộc", "gà kho",
    # "bò", "bò xào", "bò kho", "bò nướng",
    # "heo", "thịt heo", "thịt kho", "thịt nướng",
    # "tôm", "tôm chiên", "tôm rang", "tôm hấp",
    # "cá", "cá chiên", "cá kho", "cá hấp",
    # "mực", "mực xào", "mực nướng",
    # "trứng", "trứng chiên", "trứng luộc",

    # # ─────────── RAU & CHAY ───────────
    # "rau xào", "rau luộc", "rau hấp",
    # "đậu hũ", "đậu hũ chiên", "đậu hũ sốt",
    # "nấm", "nấm xào", "nấm kho",
    # "món chay", "ăn chay", "lẩu chay",

    # # ─────────── CANH & LẨU ───────────
    # "canh", "canh chua", "canh rau",
    # "canh khổ qua", "canh bí",
    # "lẩu", "lẩu thái", "lẩu hải sản", "lẩu bò", "lẩu gà",

    # # ─────────── GỎI / CUỐN ───────────
    # "gỏi", "gỏi gà", "gỏi xoài",
    # "nộm", "nộm bò khô",
    # "gỏi cuốn", "cuốn bánh tráng",

    # # ─────────── BÁNH ───────────
    # "bánh mì", "bánh xèo", "bánh cuốn",
    # "bánh bèo", "bánh khọt",
    # "bánh chưng", "bánh tét",
    # "bánh bao", "bánh giò",

    # # ─────────── ĂN VẶT ───────────
    # "ăn vặt", "bánh tráng trộn", "bánh tráng nướng",
    # "nem chua rán", "cá viên chiên",
    # "chân gà sả tắc", "ốc luộc", "ốc xào",

    # # ─────────── TRÁNG MIỆNG ───────────
    # "chè", "chè thái", "chè đậu",
    # "bánh flan", "bánh ngọt", "bánh kem",
    # "sinh tố", "nước ép", "trái cây dầm",

    # # ─────────── NGUYÊN LIỆU ───────────
    # "thịt", "cá", "tôm", "gà",
    # "rau", "củ", "nấm",
    # "trứng", "đậu",
    # "cà chua", "hành tây", "tỏi", "ớt",

    # ─────────── INTENT USER ───────────
    "món ngon mỗi ngày",
    # "món ăn gia đình",
    # "món ăn đơn giản",
    # "món ăn nhanh",
    # "món ăn tiết kiệm",
    # "món ăn cho bé",
    # "món ăn giảm cân",
]

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def polite_sleep():
    if DELAY_MIN > 0 or DELAY_MAX > 0:
        t = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(t)


def fetch(url: str, session: requests.Session, retries: int = 3) -> BeautifulSoup | None:
    """
    Tải trang và trả về BeautifulSoup.
    Tự động retry khi gặp lỗi mạng, và ngủ lâu hơn khi bị rate-limit (429).
    """
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)

            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")

            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limited (429). Ngủ {wait}s rồi thử lại...")
                time.sleep(wait)

            elif resp.status_code == 404:
                logger.debug(f"404: {url}")
                return None

            else:
                logger.warning(f"HTTP {resp.status_code}: {url}")
                time.sleep(5)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Lỗi mạng (lần {attempt + 1}/{retries}): {e}")
            time.sleep(10)

    return None


# ──────────────────────────────────────────────
# PHASE 1: THU THẬP RECIPE ID QUA TÌM KIẾM
# ──────────────────────────────────────────────

def extract_recipe_ids_from_soup(soup: BeautifulSoup) -> list[str]:
    """Trích xuất tất cả recipe ID từ trang (search hoặc category)."""
    ids = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        match = re.search(r"/vn/(?:cong-thuc|recipes)/(\d+)", href)
        if match:
            ids.append(match.group(1))
    return ids


def collect_ids_from_keyword(
    keyword: str,
    session: requests.Session,
    scraped_ids: set[str],
    mode: str,
    start_page: int,
    end_page: int,
) -> tuple[list[str], bool]:
    """
    Tìm kiếm theo keyword và thu thập recipe ID.
    Trả về: (danh sách ID mới, cờ báo đã gặp ID đã cào)
    
    Trong chế độ 'recent': Nếu gặp bất kỳ recipe ID nào đã nằm trong scraped_ids,
    ta có thể dừng cào trang tiếp theo cho từ khóa này (vì Cookpad mặc định xếp theo mới nhất).
    """
    all_ids = []
    encoded = quote(keyword)
    already_crawled_found = False

    for page in range(start_page, end_page + 1):
        if page == 1:
            url = f"{BASE_URL}/tim-kiem/{encoded}"
        else:
            url = f"{BASE_URL}/tim-kiem/{encoded}?page={page}"

        soup = fetch(url, session)
        if not soup:
            break

        ids = extract_recipe_ids_from_soup(soup)
        if not ids:
            break  # Hết kết quả

        # Kiểm tra xem có ID nào đã được cào trước đây chưa
        new_ids_in_page = []
        for rid in ids:
            if rid in scraped_ids:
                already_crawled_found = True
                if mode == "recent":
                    # Ở chế độ incremental, phát hiện ID cũ -> dừng ngay lập tức
                    logger.debug(f"  Phát hiện ID đã cào {rid} tại trang {page}. Dừng tìm kiếm '{keyword}'.")
                    break
            new_ids_in_page.append(rid)

        all_ids.extend(new_ids_in_page)
        
        if already_crawled_found and mode == "recent":
            break

        logger.debug(f"  '{keyword}' trang {page}: +{len(new_ids_in_page)} IDs")
        polite_sleep()

    unique = list(dict.fromkeys(all_ids))  # Giữ thứ tự, loại trùng
    logger.info(f"Từ khóa '{keyword}': thu thập được {len(unique)} recipe IDs mới")
    return unique, already_crawled_found


# ──────────────────────────────────────────────
# PHASE 2: PARSE TỪNG TRANG RECIPE
# ──────────────────────────────────────────────

def clean(text: str | None) -> str | None:
    """Làm sạch khoảng trắng và ký tự thừa."""
    if not text:
        return None
    return " ".join(text.split()).strip() or None


def _convert_iso_duration(iso_duration: str) -> str | None:
    """
    Chuyển đổi ISO 8601 duration (ví dụ: "PT15M", "PT1H30M") thành văn bản Việt Nam.
    """
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?$", iso_duration)
    if not match:
        return None
    
    hours, minutes = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    
    parts = []
    if hours > 0:
        parts.append(f"{hours} giờ")
    if minutes > 0:
        parts.append(f"{minutes} phút")
    
    return " ".join(parts) if parts else None


# ── Đơn vị đo lường phổ biến trong ẩm thực Việt ──────────────────────────────
_UNITS = (
    # Thể tích
    r"ml|l|lít|lit|cc|dl"
    # Khối lượng
    r"|gr?|gram|kg|kilogram|mg|mc"
    # Thể tích nấu ăn
    r"|muỗng(?:\s+(?:cà\s+phê|canh|súp))?|thìa(?:\s+(?:cà\s+phê|canh|súp))?"
    r"|đĩa|chén|bát|tô|ly|cup|tbsp|tsp|oz|lb"
    # Đếm
    r"|cái|con|quả|trái|củ|nhánh|tép|lá|bó|miếng|viên|hạt|ổ|thanh|cuộn"
    r"|lát|nắm|nhúm|vừa|đủ|chút|xíu|một\s+chút|một\s+ít|ít"
    # Kích thước / thể tích mô tả
    r"|cm|mm"
)

# Pattern bóc đầu chuỗi: số (tùy chọn) + đơn vị (tùy chọn) + dấu phân cách
_PREFIX_PATTERN = re.compile(
    r"^\s*"
    r"(?:"                          
    r"(?:\d+[\d/.,½⅓⅔¼¾⅛⅜⅝⅞]*)"  
    r"(?:\s*[-–]\s*\d+[\d/.,]*)*" 
    r"\s*"
    r"(?:" + _UNITS + r")?"       
    r"\s*"
    r")?"
    r"(?:[(\[{].*?[)\]}])?"       
    r"[\s,;:+]+"                  
    r"",
    re.IGNORECASE | re.UNICODE,
)

# Các từ phụ cần loại ở đầu NER sau khi bóc prefix
_STRIP_PREFIXES = re.compile(
    r"^\s*(?:thêm|hoặc|và|với|cùng|kèm|bao\s+gồm|gồm|như|loại|dùng|cho|để|ít|một\s+chút|một\s+ít|vài)\s+",
    re.IGNORECASE,
)

# Chú thích tùy chọn ở cuối: "(tùy thích)", "- không bắt buộc", "[optional]"
_OPTIONAL_SUFFIX = re.compile(
    r"\s*[(\[{-]\s*(?:tùy|tùy\s+(?:thích|ý|chọn)|không\s+bắt\s+buộc|optional|bỏ\s+qua|băm|nhuyễn|nướng|rang|xay)[^)\]]*[)\]]*\s*$",
    re.IGNORECASE,
)


def extract_ner(ingredient: str) -> str | None:
    """
    Bóc tách tên nguyên liệu thuần từ chuỗi ingredient đầy đủ.
    """
    if not ingredient:
        return None

    text = ingredient.strip()

    # 1. Bỏ chú thích tùy chọn ở cuối
    text = _OPTIONAL_SUFFIX.sub("", text)

    # 2. Bóc prefix số + đơn vị
    stripped = _PREFIX_PATTERN.sub("", text).strip()

    # 3. Nếu bóc xong rỗng → có thể toàn bộ là số/đơn vị → trả về text gốc
    if not stripped:
        return text.lower()

    # 4. Bóc thêm các từ liên kết thừa ở đầu
    stripped = _STRIP_PREFIXES.sub("", stripped).strip()

    # 5. Bỏ chú thích trong ngoặc ở cuối (nếu còn): "rau muống (rửa sạch)"
    stripped = re.sub(r"\s*\((?!.*\).*\().*?\)\s*$", "", stripped).strip()

    # 6. Lowercase, bỏ dấu chấm câu thừa ở cuối
    stripped = stripped.lower().rstrip(".,;:-").strip()

    return stripped if stripped else None


def parse_recipe(soup: BeautifulSoup, recipe_id: str, url: str) -> dict | None:
    """
    Parse trang recipe Cookpad VN và trả về dict có cấu trúc.
    """
    recipe: dict = {
        "id": recipe_id,
        "url": url,
        "title": None,
        "description": None,
        "images": [],
        "author": None,
        "author_location": None,
        "cook_time": None,
        "servings": None,
        "ingredients": [],
        "steps": [],
        "ner": [],
    }

    # ── Tiêu đề ──────────────────────────────
    h1 = soup.find("h1")
    recipe["title"] = clean(h1.get_text()) if h1 else None

    # ── Ảnh giới thiệu ───────────────────────
    _IMG_ATTRS = [
        "src", "data-src", "data-lazy-src", "data-original",
        "data-url", "data-image", "data-lazy", "data-echo",
        "data-bg", "data-srcset", "srcset",
    ]
    _RECIPE_IMG_RE = re.compile(r"cp(?:cdn|img)\.com/recipes/[a-fA-F0-9]+/")
    _SIZE_RE = re.compile(r"/\d+x\d+[^/]*/")

    def _strip_crop(url: str) -> str:
        m = _SIZE_RE.search(url)
        if not m:
            return url
        segment = m.group(0).strip("/")                   
        width_m = re.match(r"(\d+)x", segment)
        width = width_m.group(1) if width_m else "680"
        return _SIZE_RE.sub(f"/{width}x0cq80/", url, count=1)

    def _extract_url_from_srcset(srcset: str) -> str | None:
        for part in srcset.split(","):
            candidate = part.strip().split()[0]
            if candidate.startswith("http"):
                return candidate
        return None

    def _find_main_image_url(soup: BeautifulSoup) -> str | None:
        og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
        if og:
            content = og.get("content", "")
            if content and ("cpcdn.com" in content or "cpimg.com" in content):
                return _strip_crop(content)

        for img in soup.find_all("img"):
            for attr in _IMG_ATTRS:
                val = img.get(attr, "") or ""
                if attr in ("srcset", "data-srcset") and val:
                    val = _extract_url_from_srcset(val) or ""
                if val and _RECIPE_IMG_RE.search(val):
                    return _strip_crop(val)

        for tag in soup.find_all(True, style=True):
            style = tag.get("style", "")
            m = re.search(
                r"url\(['\"]?(https?://[^'\")\s]+cp(?:cdn|img)\.com/recipes/[^'\")\s]+)['\"]?\)",
                style,
            )
            if m:
                return _strip_crop(m.group(1))

        img_link = soup.find("a", href=re.compile(r"/(?:vn/)?recipe/images/([a-fA-F0-9]+)"))
        if img_link:
            m = re.search(r"/recipe/images/([a-fA-F0-9]+)", img_link["href"])
            if m:
                h = m.group(1)
                return f"https://img-global.cpcdn.com/recipes/{h}/680x0cq80/photo.webp"

        return None

    main_image = _find_main_image_url(soup)
    recipe["images"] = [main_image] if main_image else []

    # ── Thời gian và Khẩu phần từ Structured Data (JSON-LD) ───
    try:
        script_tag = soup.find("script", type="application/ld+json")
        if script_tag:
            schema = json.loads(script_tag.string)
            if "cookTime" in schema and schema["cookTime"]:
                cook_time_iso = schema["cookTime"]
                recipe["cook_time"] = _convert_iso_duration(cook_time_iso) or recipe["cook_time"]
            if "recipeYield" in schema and schema["recipeYield"]:
                recipe["servings"] = str(schema["recipeYield"])
    except Exception:
        pass

    # ── Tác giả ──────────────────────────────
    author_link = soup.find("a", href=re.compile(r"/vn/nguoi-su-dung/\d+"))
    if author_link:
        recipe["author"] = clean(author_link.get_text())
        location_el = author_link.find_next(string=re.compile(r"Việt Nam|Hà Nội|TP\.|Hồ Chí Minh|Đà Nẵng"))
        if location_el:
            recipe["author_location"] = clean(str(location_el))

    # ── Mô tả ────────────────────────────────
    if h1:
        next_el = h1.find_next_sibling()
        while next_el:
            tag = next_el.name
            if tag in ["h2", "h3"]:
                break
            if tag in ["p", "div"]:
                text = clean(next_el.get_text())
                text = re.sub(r"\s*[Xx]em\s+[Tt]h[eê]m\s*$", "", text).strip()
                if text and len(text) > 20:
                    recipe["description"] = text
                    break
            next_el = next_el.find_next_sibling()

    # ── Thời gian nấu và Khẩu phần ──────────────────────────
    if not recipe["cook_time"]:
        for el in soup.find_all(string=re.compile(r"\d+.*?(?:phút|giờ)", re.IGNORECASE)):
            text = clean(str(el).strip())
            if text and len(text) < 30 and re.search(r"^\d+\s*(?:giờ|phút)", text):
                recipe["cook_time"] = text
                break
        
        if not recipe["cook_time"]:
            all_text = soup.get_text()
            match = re.search(r"(\d+\s*(?:giờ[^a-z]*)?(?:\d+\s+)?phút)", all_text)
            if match:
                recipe["cook_time"] = match.group(0).strip()

    if not recipe["servings"]:
        for el in soup.find_all(string=re.compile(r"\d+.*?(?:người|phần ăn|khẩu phần)", re.IGNORECASE)):
            text = clean(str(el).strip())
            if text and len(text) < 30 and re.search(r"^\d+\s*(?:người|phần ăn|khẩu phần)", text):
                recipe["servings"] = text
                break

    # ── Nguyên liệu ──────────────────────────
    ing_heading = soup.find(
        lambda t: t.name in ["h2", "h3"] and "Nguy" in t.get_text()
    )
    if ing_heading:
        ing_list = ing_heading.find_next(["ul", "ol"])
        if ing_list:
            for li in ing_list.find_all("li", recursive=False):
                text = clean(li.get_text())
                if text:
                    recipe["ingredients"].append(text)

    # ── Các bước làm ─────────────────────────
    step_heading = soup.find(
        lambda t: t.name in ["h2", "h3"] and "Hướng dẫn" in t.get_text()
    )
    if step_heading:
        step_list = step_heading.find_next(["ol", "ul"])
        if step_list:
            for li in step_list.find_all("li", recursive=False):
                img_urls = []
                seen_urls = set()

                _STEP_IMG_ATTRS = [
                    "src", "data-src", "data-lazy-src", "data-original",
                    "data-url", "data-image", "data-lazy", "data-echo",
                    "srcset", "data-srcset",
                ]
                
                for img_tag in li.find_all("img"):
                    found_url = False
                    for attr in _STEP_IMG_ATTRS:
                        if found_url:
                            break
                        val = img_tag.get(attr, "") or ""
                        if attr in ("srcset", "data-srcset") and val:
                            val = _extract_url_from_srcset(val) or ""
                        if val and val.startswith("http"):
                            img_url = _SIZE_RE.sub("/640x480cq80/", val, count=1)
                            match_hash = re.search(r"/steps/([a-f0-9]+)/", img_url)
                            url_hash = match_hash.group(1) if match_hash else img_url
                            if url_hash not in seen_urls:
                                img_urls.append(img_url)
                                seen_urls.add(url_hash)
                            found_url = True

                if not img_urls:
                    for tag in li.find_all(True, style=True):
                        style = tag.get("style", "")
                        m = re.search(r"url\(['\"]?(https?://[^'\")\s]+)['\"]?\)", style)
                        if m:
                            img_url = _SIZE_RE.sub("/640x480cq80/", m.group(1), count=1)
                            match_hash = re.search(r"/steps/([a-f0-9]+)/", img_url)
                            url_hash = match_hash.group(1) if match_hash else img_url
                            if url_hash not in seen_urls:
                                img_urls.append(img_url)
                                seen_urls.add(url_hash)

                if not img_urls:
                    step_links = li.find_all(
                        "a", href=re.compile(r"/step_attachment/images/([a-fA-F0-9]+)")
                    )
                    for step_link in step_links:
                        m = re.search(
                            r"/step_attachment/images/([a-fA-F0-9]+)",
                            step_link["href"],
                        )
                        if m:
                            h = m.group(1)
                            if h not in seen_urls:
                                img_url = (
                                    f"https://img-global.cpimg.com/steps/{h}"
                                    f"/640x480cq80/photo.jpg"
                                )
                                img_urls.append(img_url)
                                seen_urls.add(h)

                for _img in li.find_all("img"):
                    _img.decompose()

                text = clean(li.get_text())
                if text:
                    recipe["steps"].append({
                        "text": text,
                        "images": img_urls,
                    })

    # ── NER: tên nguyên liệu thuần ───────────
    seen: set[str] = set()
    for ing in recipe["ingredients"]:
        name = extract_ner(ing)
        if name and name not in seen:
            recipe["ner"].append(name)
            seen.add(name)

    # ── Validate ─────────────────────────────
    if not recipe["title"]:
        return None
    if not recipe["ingredients"] and not recipe["steps"]:
        return None

    return recipe


# ──────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────

def load_checkpoint(checkpoint_file: Path) -> dict:
    if checkpoint_file.exists():
        with open(checkpoint_file, encoding="utf-8") as f:
            return json.load(f)
    return {"searched_keywords": [], "all_recipe_ids": [], "scraped_ids": []}


def save_checkpoint(cp: dict, checkpoint_file: Path):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cookpad Recipe Crawler.")
    parser.add_argument(
        "--mode",
        choices=["recent", "historical"],
        default="recent",
        help="Chế độ cào: 'recent' (chỉ cào các món mới nhất, dừng khi gặp ID cũ) hoặc 'historical' (cào toàn bộ trang được cấu hình)."
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Trang bắt đầu tìm kiếm (Mặc định: 1 cho recent, 31 cho historical)."
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Trang kết thúc tìm kiếm (Mặc định: 3 cho recent, 60 cho historical)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Thư mục chứa kết quả và log."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Số lượng luồng cào đồng thời."
    )
    
    args = parser.parse_args()

    # Xác định tham số phân trang dựa theo chế độ
    if args.mode == "recent":
        start_page = args.start_page if args.start_page is not None else 1
        end_page = args.end_page if args.end_page is not None else 3
    else:  # historical
        start_page = args.start_page if args.start_page is not None else 31
        end_page = args.end_page if args.end_page is not None else 60

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_file = output_dir / "checkpoint.json"
    output_file = output_dir / "recipes.jsonl"
    failed_file = output_dir / "failed_ids.txt"

    # Cấu hình lại log file sau khi tạo thư mục
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    logger.addHandler(logging.FileHandler(output_dir / "scraper.log", encoding="utf-8"))

    cp = load_checkpoint(checkpoint_file)
    
    # Ở chế độ cào 'recent' định kỳ:
    # Ta KHÔNG dùng searched_keywords từ checkpoint cũ để bỏ qua các từ khóa,
    # bởi vì mỗi phiên chạy định kỳ đều phải quét qua mọi từ khóa để tìm món mới.
    if args.mode == "recent":
        searched_keywords = set()
    else:
        searched_keywords = set(cp.get("searched_keywords", []))

    all_recipe_ids = cp.get("all_recipe_ids", [])
    scraped_ids = set(cp.get("scraped_ids", []))
    id_set = set(all_recipe_ids)

    session = requests.Session()
    session.headers.update(HEADERS)

    # ── PHASE 1: Thu thập ID ──────────────────
    logger.info("=" * 50)
    logger.info(f"PHASE 1: Thu thập Recipe ID mới nhất (Chế độ: {args.mode.upper()}, Trang {start_page} -> {end_page})")
    logger.info("=" * 50)

    for keyword in VIETNAMESE_KEYWORDS:
        if keyword in searched_keywords:
            continue
        
        ids, hit_crawled = collect_ids_from_keyword(
            keyword, session, scraped_ids, args.mode, start_page, end_page
        )
        new_ids = [i for i in ids if i not in id_set]
        id_set.update(new_ids)
        all_recipe_ids.extend(new_ids)
        searched_keywords.add(keyword)

        # Lưu checkpoint sau mỗi keyword
        cp["searched_keywords"] = list(searched_keywords)
        cp["all_recipe_ids"] = all_recipe_ids
        save_checkpoint(cp, checkpoint_file)

        polite_sleep()

    logger.info(f"Tổng số Recipe ID hiện có trong danh sách: {len(all_recipe_ids)}")

    # ── PHASE 2: Scrape từng recipe ───────────
    to_scrape = [rid for rid in all_recipe_ids if rid not in scraped_ids]
    if MAX_RECIPES > 0:
        to_scrape = to_scrape[:MAX_RECIPES]

    if not to_scrape:
        logger.info("Không có recipe mới nào cần cào thêm.")
        logger.info("=" * 50)
        return

    logger.info("=" * 50)
    logger.info(f"PHASE 2: Tiến hành cào {len(to_scrape)} recipes (sử dụng {args.workers} workers)")
    logger.info("=" * 50)

    failed_ids = []
    file_lock = Lock()
    state_lock = Lock()
    
    def scrape_single_recipe(recipe_id: str) -> tuple[str, dict | None, bool]:
        url = f"{BASE_URL}/cong-thuc/{recipe_id}"
        thread_session = requests.Session()
        thread_session.headers.update(HEADERS)
        try:
            soup = fetch(url, thread_session)
            if soup:
                recipe = parse_recipe(soup, recipe_id, url)
                if recipe:
                    return (recipe_id, recipe, True)
            return (recipe_id, None, False)
        except Exception as e:
            logger.error(f"Lỗi khi cào recipe {recipe_id}: {e}")
            return (recipe_id, None, False)
        finally:
            thread_session.close()

    nonlocal_vars = {'success': 0, 'failed': 0}
    
    with (
        open(output_file, "a", encoding="utf-8") as out_f,
        open(failed_file, "a", encoding="utf-8") as fail_f,
        tqdm(total=len(to_scrape), desc="Đang cào", unit="recipe") as pbar,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures = {
            executor.submit(scrape_single_recipe, recipe_id): recipe_id 
            for recipe_id in to_scrape
        }
        
        for future in as_completed(futures):
            recipe_id, recipe_data, success = future.result()
            
            with file_lock:
                if success and recipe_data:
                    out_f.write(json.dumps(recipe_data, ensure_ascii=False) + "\n")
                    out_f.flush()
                    nonlocal_vars['success'] += 1
                else:
                    fail_f.write(recipe_id + "\n")
                    fail_f.flush()
                    nonlocal_vars['failed'] += 1
                    failed_ids.append(recipe_id)
                
                with state_lock:
                    scraped_ids.add(recipe_id)
                    if len(scraped_ids) % 100 == 0:
                        cp["scraped_ids"] = list(scraped_ids)
                        save_checkpoint(cp, checkpoint_file)
            
            pbar.update(1)
            pbar.set_postfix({"ok": nonlocal_vars['success'], "fail": nonlocal_vars['failed']})
            polite_sleep()

    success_count = nonlocal_vars['success']
    cp["scraped_ids"] = list(scraped_ids)
    save_checkpoint(cp, checkpoint_file)

    logger.info("=" * 50)
    logger.info(f"HOÀN THÀNH PHIÊN CÀO!")
    logger.info(f"  Thành công : {success_count} recipes")
    logger.info(f"  Thất bại   : {len(failed_ids)} IDs")
    logger.info(f"  File output: {output_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
