import argparse
import sys
import pandas as pd
import requests
from urllib.parse import urlparse, unquote
from functools import lru_cache

UA = {"User-Agent": "commons-wiki-desc/1.0"}
_session = requests.Session()
_session.headers.update(UA)

# ---------------- Commons → Wikidata → Wikipedia ----------------

def commons_url_to_title(url: str) -> str:
    path = urlparse(str(url)).path
    return unquote(path.split("/wiki/")[1])

@lru_cache(maxsize=4096)
def _commons_pageprops(title: str) -> dict:
    r = _session.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action":"query","format":"json","prop":"pageprops","titles":title,
            "redirects":1,"converttitles":1
        },
        timeout=15
    ).json()
    return next(iter(r["query"]["pages"].values())).get("pageprops", {})

def _resolve_commons_redirect(title: str) -> str:
    """카테고리 소프트 리다이렉트 등 처리."""
    props = _commons_pageprops(title)
    if props.get("wikibase_item"):
        return title

    target = props.get("categoryredirect")
    if target:
        if not target.startswith("Category:"):
            target = "Category:" + target
        return _resolve_commons_redirect(target)

    r = _session.get(
        "https://commons.wikimedia.org/w/api.php",
        params={"action":"parse","format":"json","page":title,"prop":"links"},
        timeout=15
    ).json()
    for l in r.get("parse", {}).get("links", []):
        if l.get("ns") == 14 and "exists" in l:
            t = l["*"]
            guess = t if t.startswith("Category:") else "Category:" + t
            return _resolve_commons_redirect(guess)
    return title

def _normalize_category_ns(title: str) -> str:
    if title.startswith("Category:Category:"):
        title = title.replace("Category:Category:", "Category:", 1)
    return title.strip()

def _first_entity_id(claims: dict, pid: str):
    arr = claims.get(pid, [])
    if not arr: return None
    dv = arr[0]["mainsnak"].get("datavalue")
    if dv and dv.get("type") == "wikibase-entityid":
        return "Q" + str(dv["value"]["numeric-id"])
    return None

def _commons_title_to_qid(title: str) -> str:
    props = _commons_pageprops(title)
    qid = props.get("wikibase_item")
    if qid:
        return qid
    r = _session.get(
        "https://www.wikidata.org/w/api.php",
        params={"action":"wbgetentities","sites":"commonswiki","titles":title,
                "props":"","format":"json"},
        timeout=15
    ).json()
    ents = r.get("entities", {})
    for k in ents:
        if k.startswith("Q"):
            return k
    raise RuntimeError(f"QID not found for {title}")

@lru_cache(maxsize=4096)
def _get_entity(qid: str) -> dict:
    r = _session.get(
        f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
        timeout=15
    ).json()
    return r["entities"][qid]

def _topic_qid_if_category(qid: str) -> str:
    ent = _get_entity(qid)
    claims = ent.get("claims", {})
    P31 = claims.get("P31", [])
    is_cat = any(
        s.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q4167836"
        for s in P31 if s.get("mainsnak", {}).get("datavalue")
    )
    if is_cat:
        q_topic = _first_entity_id(claims, "P301")  # category's main topic
        if q_topic:
            return q_topic
    return qid

def commons_to_wikipedia_url(commons_url: str, wiki_lang: str = "en") -> str | None:
    """Commons 카테고리 URL → (리다이렉트/정규화) → QID → (본 주제) → Wikipedia sitelink URL"""
    try:
        title = commons_url_to_title(commons_url)
    except Exception:
        return None
    title = _resolve_commons_redirect(title)
    title = _normalize_category_ns(title)
    qid = _commons_title_to_qid(title)
    qid = _topic_qid_if_category(qid)
    sitelinks = _get_entity(qid).get("sitelinks", {})
    key = f"{wiki_lang}wiki"
    if key in sitelinks:
        return sitelinks[key]["url"]
    # 폴백: en/kowiki 등 다른 언어라도 하나 반환
    for alt in ("enwiki", "kowiki", "dewiki", "frwiki"):
        if alt in sitelinks:
            return sitelinks[alt]["url"]
    return None

# ---------------- Wikipedia lead(참조번호 제거) ----------------

def wikipedia_intro_clean(wiki_url: str) -> str | None:
    """
    MediaWiki Extracts API로 리드 섹션 전체(여러 문단)를
    참조번호 없이 순수 텍스트로 반환.
    """
    try:
        netloc = urlparse(wiki_url).netloc  # e.g., en.wikipedia.org
        lang = netloc.split(".")[0]
        title = unquote(urlparse(wiki_url).path.split("/wiki/")[1])
        r = _session.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,       # 리드(섹션 0)
                "explaintext": 1,   # HTML 제거
                "redirects": 1,
                "titles": title,
                "format": "json"
            },
            timeout=15
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return page.get("extract") or None  # 여러 문단은 \n\n로 구분
    except Exception:
        return None

# ---------------- DataFrame 처리 ----------------

def safe_resolve_wiki_url(commons_url: str, wiki_lang="en") -> str | None:
    try:
        if not isinstance(commons_url, str) or not commons_url:
            return None
        return commons_to_wikipedia_url(commons_url, wiki_lang=wiki_lang)
    except Exception as e:
        print(f"[WARN] [url] {commons_url} -> {e}", file=sys.stderr)
        return None

def safe_fetch_description(wiki_url: str) -> str | None:
    try:
        if not isinstance(wiki_url, str) or not wiki_url:
            return None
        return wikipedia_intro_clean(wiki_url)
    except Exception as e:
        print(f"[WARN] [description] {wiki_url} -> {e}", file=sys.stderr)
        return None

def enrich_with_url_and_description(df: pd.DataFrame, url_col: str = "category",
                                    wiki_lang: str = "en", show_progress: bool = True) -> pd.DataFrame:
    out = df.copy()
    if show_progress:
        try:
            from tqdm import tqdm
            # 1) Wikipedia URL 생성
            try:
                from tqdm.auto import tqdm as auto_tqdm
                auto_tqdm.pandas(desc="Resolving Wikipedia URL")
            except Exception:
                tqdm.pandas(desc="Resolving Wikipedia URL")
            out["url"] = out[url_col].progress_map(lambda u: safe_resolve_wiki_url(u, wiki_lang=wiki_lang))

            # 2) description(리드 문단, 참조번호 제거)
            try:
                from tqdm.auto import tqdm as auto_tqdm2
                auto_tqdm2.pandas(desc="Fetching Wikipedia lead (clean)")
            except Exception:
                tqdm.pandas(desc="Fetching Wikipedia lead (clean)")
            out["description"] = out["url"].progress_map(safe_fetch_description)
        except Exception as e:
            print(f"[INFO] tqdm 사용 불가(일반 처리로 대체): {e}", file=sys.stderr)
            out["url"] = out[url_col].map(lambda u: safe_resolve_wiki_url(u, wiki_lang=wiki_lang))
            out["description"] = out["url"].map(safe_fetch_description)
    else:
        out["url"] = out[url_col].map(lambda u: safe_resolve_wiki_url(u, wiki_lang=wiki_lang))
        out["description"] = out["url"].map(safe_fetch_description)
    return out

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Add 'url' (Wikipedia link) and 'description' (lead text without references) to a CSV using Commons category URLs."
    )
    ap.add_argument("-i", "--input", required=True, help="입력 CSV 경로")
    ap.add_argument("-o", "--output", required=True, help="출력 CSV 경로")
    ap.add_argument("-c", "--url-col", default="category", help="Commons URL이 들어있는 컬럼명 (기본: category)")
    ap.add_argument("--wiki-lang", default="en", help="Wikipedia 언어 (기본: en)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--progress", dest="progress", action="store_true", help="진행막대 표시(기본)")
    g.add_argument("--no-progress", dest="progress", action="store_false", help="진행막대 끄기")
    ap.set_defaults(progress=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.url_col not in df.columns:
        print(f"[ERROR] 입력 CSV에 '{args.url_col}' 컬럼이 없습니다. 실제 컬럼들: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    out = enrich_with_url_and_description(
        df, url_col=args.url_col, wiki_lang=args.wiki_lang, show_progress=args.progress
    )
    out.drop(columns=["hierarchical_label", "natural_or_human_made"], inplace=True, errors="ignore")
    out.to_csv(args.output, index=False)
    print(f"[OK] Saved with 'url' and 'description' columns -> {args.output}")

if __name__ == "__main__":
    main()
