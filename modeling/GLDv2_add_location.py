import argparse
import sys
import pandas as pd
import requests
from urllib.parse import urlparse, unquote

UA = {"User-Agent": "commons-location-extractor/1.0"}
_session = requests.Session()
_session.headers.update(UA)

def commons_url_to_title(url: str) -> str:
    path = urlparse(url).path
    return unquote(path.split("/wiki/")[1])

def _commons_pageprops(title: str) -> dict:
    r = _session.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action":"query","format":"json","prop":"pageprops","titles":title,
            "redirects":1,
            "converttitles":1
        },
        timeout=15
    ).json()
    return next(iter(r["query"]["pages"].values())).get("pageprops", {})

def _resolve_commons_redirect(title: str) -> str:
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

def _get_entity(qid: str) -> dict:
    r = _session.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
                     timeout=15).json()
    return r["entities"][qid]

def _first_entity_id(claims: dict, pid: str):
    arr = claims.get(pid, [])
    if not arr: return None
    dv = arr[0]["mainsnak"].get("datavalue")
    if dv and dv["type"] == "wikibase-entityid":
        return "Q" + str(dv["value"]["numeric-id"])
    return None

def _labels(qids, lang="en"):
    if not qids: return {}
    r = _session.get(
        "https://www.wikidata.org/w/api.php",
        params={"action":"wbgetentities","ids":"|".join(qids),
                "props":"labels","languages":f"{lang}|en","format":"json"},
        timeout=15
    ).json()
    out = {}
    for q in qids:
        ent = r["entities"].get(q, {}).get("labels", {})
        lbl = ent.get(lang) or ent.get("en")
        if lbl: out[q] = lbl["value"]
    return out

def _topic_qid_if_category(qid: str) -> str:
    ent = _get_entity(qid)
    claims = ent.get("claims", {})
    P31 = claims.get("P31", [])
    is_cat = any(s.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q4167836"
                 for s in P31 if s.get("mainsnak", {}).get("datavalue"))
    if is_cat:
        q_topic = _first_entity_id(claims, "P301")  # category's main topic
        if q_topic:
            return q_topic
    return qid

def _build_location_chain(qid: str, lang="en"):
    ent = _get_entity(qid)
    claims = ent["claims"]
    chain = []
    for pid in ("P276","P131","P706","P17"):
        q = _first_entity_id(claims, pid)
        if not q: continue
        if pid == "P131":
            cur = q
            while cur:
                if cur not in chain: chain.append(cur)
                e = _get_entity(cur)
                cur = _first_entity_id(e["claims"], "P131")
        else:
            if q not in chain: chain.append(q)
    names = [_labels(chain, lang=lang).get(q, q) for q in chain]
    return names

def _normalize_category_ns(title: str) -> str:
    if title.startswith("Category:Category:"):
        title = title.replace("Category:Category:", "Category:", 1)
    return title.strip()

def extract_location_from_commons(url: str, lang="en") -> str | None:
    title = commons_url_to_title(url)
    title = _resolve_commons_redirect(title)
    title = _normalize_category_ns(title)
    qid = _commons_title_to_qid(title)
    qid = _topic_qid_if_category(qid)
    names = _build_location_chain(qid, lang=lang)
    if not names:
        return None
    return ", ".join(names)

def safe_extract(url: str, lang="en") -> str | None:
    try:
        return extract_location_from_commons(url, lang=lang)
    except Exception as e:
        print(f"[WARN] {url} -> {e}", file=sys.stderr)
        return None

def enrich_with_location(df: pd.DataFrame, url_col: str = "category", lang: str = "en",
                         show_progress: bool = True) -> pd.DataFrame:
    out = df.copy()
    if show_progress:
        try:
            from tqdm import tqdm
            # tqdm with pandas integration
            try:
                from tqdm.auto import tqdm as auto_tqdm  # nice Jupyter/tty behavior
                auto_tqdm.pandas(desc="Resolving Commons locations")
            except Exception:
                tqdm.pandas(desc="Resolving Commons locations")
            out["location"] = out[url_col].progress_map(lambda u: safe_extract(u, lang=lang))
        except Exception as e:
            print(f"[INFO] tqdm 사용 불가(일반 처리로 대체): {e}", file=sys.stderr)
            out["location"] = out[url_col].map(lambda u: safe_extract(u, lang=lang))
    else:
        out["location"] = out[url_col].map(lambda u: safe_extract(u, lang=lang))
    return out

def main():
    ap = argparse.ArgumentParser(description="Add Commons-derived 'location' to a CSV with a Commons URL column.")
    ap.add_argument("-i", "--input", required=True, help="입력 CSV 경로 (train_hierarchical_df)")
    ap.add_argument("-o", "--output", required=True, help="출력 CSV 경로 (train_location_df)")
    ap.add_argument("-c", "--url-col", default="category", help="Commons URL이 들어있는 컬럼명 (기본: category)")
    ap.add_argument("--lang", default="en", help="라벨 언어 (예: en, ko; 기본: en)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--progress", dest="progress", action="store_true", help="진행막대 표시(기본)")
    g.add_argument("--no-progress", dest="progress", action="store_false", help="진행막대 끄기")
    ap.set_defaults(progress=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.url_col not in df.columns:
        print(f"[ERROR] 입력 CSV에 '{args.url_col}' 컬럼이 없습니다. 실제 컬럼들: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    out = enrich_with_location(df, url_col=args.url_col, lang=args.lang, show_progress=args.progress)
    out.to_csv(args.output, index=False)
    print(f"[OK] Saved with 'location' column -> {args.output}")

if __name__ == "__main__":
    main()
