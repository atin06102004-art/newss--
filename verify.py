"""
verify.py
Web-based evidence/verification layer for the Fake News Detector.

Uses Tavily to search the live web for the same claim the user entered,
then compares the retrieved evidence against the ML model's prediction
to produce a final, corrected verdict.

Get a free Tavily API key at https://tavily.com and set it as an
environment variable named TAVILY_API_KEY (see README for setup).
"""

import os
import re
from dataclasses import dataclass, field

from tavily import TavilyClient

# A short list of outlets we treat as stronger evidence when they show up
# in results. This is just used to order/label sources — Tavily itself
# already ranks by relevance.
TRUSTED_DOMAINS = [
    # Global wire services & major international outlets
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "aljazeera.com", "cnn.com", "cbsnews.com",
    "nbcnews.com", "abcnews.com", "washingtonpost.com", "nytimes.com",
    "foxnews.com", "wsj.com", "bloomberg.com", "cnbc.com", "usatoday.com",
    "time.com", "newsweek.com", "economist.com", "afp.com",
    # Canada
    "cbc.ca", "ctvnews.ca", "globalnews.ca",
    # Middle East / Israel
    "jpost.com", "timesofisrael.com", "haaretz.com", "arabnews.com",
    # India
    "thehindu.com", "ndtv.com", "indianexpress.com", "hindustantimes.com",
    "timesofindia.indiatimes.com", "livemint.com",
    # Local / regional US news networks
    "spectrumlocalnews.com", "wng.org",
    # Government / institutional / specialist
    "gov.in", "who.int", "un.org", "espncricinfo.com", "icc-cricket.com",
    "whitehouse.gov",
]

# Words that don't help decide topical relevance — stripped before comparing.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of",
    "to", "for", "and", "or", "that", "this", "with", "at", "by",
    "has", "have", "had", "will", "be", "as", "it", "its", "new",
}

# Minimum overlap score (0-1) for a search result to count as "relevant"
# to the claim. Tune this if you find matches being missed or too loose.
RELEVANCE_THRESHOLD = 0.35

# Minimum overlap score (0-1) for a SINGLE trusted source, on its own,
# to be enough to confirm VERIFIED REAL without a second source.
STRONG_MATCH_THRESHOLD = 0.6

# --- Source-count verification rule -----------------------------------
# A claim is only marked VERIFIED REAL when at least this many
# *independent* (distinct-domain) sources report on it. Fewer than this
# is marked UNVERIFIED, not FAKE — absence of enough coverage doesn't
# mean the claim is false, just that it isn't confirmed yet.
# Set to 3 by default (i.e. "3-4 sources"); bump to 4 if you want a
# stricter bar.
MIN_SOURCES_FOR_VERIFIED = 3


@dataclass
class Evidence:
    title: str
    url: str
    domain: str
    snippet: str
    score: float
    is_trusted: bool = False
    overlap_score: float = 0.0  # debug: how well this result matches the claim (0-1)


@dataclass
class VerificationResult:
    query: str
    evidence: list = field(default_factory=list)
    verdict: str = "UNVERIFIED"          # VERIFIED_REAL | LIKELY_FAKE | UNVERIFIED
    verdict_label: str = "🟡 UNVERIFIED"
    reason: str = ""
    matched_sources: list = field(default_factory=list)


def _get_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Get a free key at https://tavily.com "
            "and set it as an environment variable (see README)."
        )
    return TavilyClient(api_key=api_key)


def _domain_from_url(url: str) -> str:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return match.group(1) if match else url


def search_web(claim: str, max_results: int = 6) -> list:
    """
    Search the live web for the same claim using Tavily.
    Returns a list of Evidence objects.
    """
    client = _get_client()

    response = client.search(
        query=claim,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
    )

    results = []
    for r in response.get("results", []):
        domain = _domain_from_url(r.get("url", ""))
        results.append(
            Evidence(
                title=r.get("title", ""),
                url=r.get("url", ""),
                domain=domain,
                snippet=r.get("content", "")[:300],
                score=r.get("score", 0.0),
                is_trusted=any(domain.endswith(d) for d in TRUSTED_DOMAINS),
            )
        )

    # Trusted sources first, then by Tavily relevance score.
    results.sort(key=lambda e: (not e.is_trusted, -e.score))
    return results


def _normalize_words(words):
    """
    Strip stopwords and apply light suffix-stemming so word-form
    differences (arrives/arrived, sanctions/sanctioned) still match.
    """
    out = set()
    for w in words:
        if w in STOPWORDS:
            continue
        stemmed = w
        for suf in ("ing", "ed", "es", "s"):
            if stemmed.endswith(suf) and len(stemmed) - len(suf) >= 3:
                stemmed = stemmed[: -len(suf)]
                break
        out.add(stemmed)
    return out


def _keyword_overlap(claim: str, text: str) -> float:
    """
    Lexical-overlap score between the claim and a piece of evidence text.
    Scored against the SMALLER of the two word sets (not just the claim),
    so long/multi-part claims aren't unfairly penalized when an article
    only echoes part of them. Light stemming handles simple word-form
    differences. This is a cheap heuristic, not semantic matching — good
    enough to flag whether a result is actually about the claim.
    """
    claim_words = _normalize_words(re.findall(r"[a-z0-9]+", claim.lower()))
    text_words = _normalize_words(re.findall(r"[a-z0-9]+", text.lower()))

    if not claim_words or not text_words:
        return 0.0

    overlap = len(claim_words & text_words)
    denom = min(len(claim_words), len(text_words))
    return overlap / denom


def verify_claim(claim: str, ml_label: str, ml_confidence: float, max_results: int = 6) -> VerificationResult:
    """
    Search the web for the claim and reconcile the evidence with the
    ML model's prediction to produce one of three final verdicts:

      🟢 VERIFIED REAL  - at least MIN_SOURCES_FOR_VERIFIED independent
                           sources report on the claim
      🔴 LIKELY FAKE    - no credible supporting evidence found AND the ML
                           model is confident it's fake
      🟡 UNVERIFIED     - fewer than MIN_SOURCES_FOR_VERIFIED sources found
                           (absence of enough coverage does NOT prove the
                           claim is false)

    The verdict is now driven primarily by *source count*: a claim needs
    corroboration from several independent outlets (distinct domains) to
    be called verified. Anything below that bar is UNVERIFIED, not FAKE —
    only a total absence of coverage combined with a confident ML "FAKE"
    prediction pushes the verdict to LIKELY FAKE.

    ml_label: "REAL" or "FAKE" (the raw ML model prediction)
    ml_confidence: 0-100
    """
    try:
        evidence = search_web(claim, max_results=max_results)
    except RuntimeError as e:
        return VerificationResult(
            query=claim,
            evidence=[],
            verdict="UNVERIFIED",
            verdict_label="🟡 UNVERIFIED",
            reason=str(e),
        )

    # Score how relevant each result actually is to the claim text,
    # not just whatever Tavily returned. Store the score on each item too
    # (debug: visible in the UI) so thresholds can be tuned from real data.
    for e in evidence:
        e.overlap_score = _keyword_overlap(claim, e.title + " " + e.snippet)

    relevant = [e for e in evidence if e.overlap_score >= RELEVANCE_THRESHOLD]

    # Independent confirmation requires distinct domains — two articles
    # from the same outlet aren't two separate confirmations.
    relevant_domains = {e.domain for e in relevant}
    trusted_relevant = [e for e in relevant if e.is_trusted]
    trusted_domains_found = {e.domain for e in trusted_relevant}

    matched_sources = sorted(trusted_domains_found or relevant_domains)[:5]
    source_count = len(relevant_domains)

    ml_says_fake = ml_label == "FAKE"
    ml_says_real = ml_label == "REAL"
    ml_is_confident = ml_confidence >= 70

    def _ml_agreement_note(evidence_points_real: bool) -> str:
        """Short clause describing how the ML prediction relates to the
        web evidence, appended to the main reason string."""
        if evidence_points_real and ml_says_real:
            return f" This matches the ML model's own prediction (REAL, {ml_confidence:.1f}% confidence)."
        if evidence_points_real and ml_says_fake:
            return (
                f" Note: the ML model predicted FAKE ({ml_confidence:.1f}% confidence) based on writing "
                "style alone, but the live web evidence contradicts it — the web evidence is treated "
                "as more reliable here."
            )
        if (not evidence_points_real) and ml_says_fake:
            return f" This matches the ML model's own prediction (FAKE, {ml_confidence:.1f}% confidence)."
        if (not evidence_points_real) and ml_says_real:
            return (
                f" Note: the ML model predicted REAL ({ml_confidence:.1f}% confidence), but not enough "
                "independent sources confirm this claim yet."
            )
        return ""

    # --- Decision logic: driven primarily by independent source count ---
    if source_count >= MIN_SOURCES_FOR_VERIFIED:
        # Enough independent outlets are reporting on this — call it verified.
        verdict = "VERIFIED_REAL"
        verdict_label = "🟢 VERIFIED REAL"
        reason = (
            f"Found {source_count} independent sources "
            f"({', '.join(matched_sources)}) reporting on this claim — meets the "
            f"{MIN_SOURCES_FOR_VERIFIED}+ independent-source threshold for verification."
            + _ml_agreement_note(evidence_points_real=True)
        )

    elif source_count >= 1:
        # Some coverage exists, but not enough distinct outlets yet to
        # clear the verification bar. One special case: a single very
        # strong, highly-relevant match from a major trusted wire service
        # (e.g. Reuters/AP breaking a story before anyone else has) can
        # still be called verified on its own.
        single = max(relevant, key=lambda e: e.overlap_score)
        if single.is_trusted and single.overlap_score >= STRONG_MATCH_THRESHOLD:
            verdict = "VERIFIED_REAL"
            verdict_label = "🟢 VERIFIED REAL"
            reason = (
                f"Confirmed by one major trusted source with a strong match: {single.domain}. "
                "(Below the usual multi-source threshold, but the match quality and source "
                "reliability are high enough on their own.)"
                + _ml_agreement_note(evidence_points_real=True)
            )
        else:
            verdict = "UNVERIFIED"
            verdict_label = "🟡 UNVERIFIED"
            reason = (
                f"Only {source_count} independent source(s) found "
                f"({', '.join(matched_sources)}) — below the "
                f"{MIN_SOURCES_FOR_VERIFIED}+ needed to confirm. Not enough corroboration yet."
                + _ml_agreement_note(evidence_points_real=False)
            )

    else:
        # No relevant coverage found at all — the ML prediction becomes
        # the primary signal.
        if ml_says_fake and ml_is_confident:
            verdict = "LIKELY_FAKE"
            verdict_label = "🔴 LIKELY FAKE"
            reason = (
                "No credible sources report this claim, and the ML model "
                f"flagged it as fake with {ml_confidence:.1f}% confidence."
            )
        elif ml_says_real and ml_is_confident:
            verdict = "UNVERIFIED"
            verdict_label = "🟡 UNVERIFIED"
            reason = (
                "No matching articles found on the web. This does not prove the claim "
                "is false — it may simply be too recent, too niche, or not yet covered. "
                f"The ML model leans REAL ({ml_confidence:.1f}% confidence) based on writing "
                "style, but that alone isn't confirmation."
            )
        else:
            verdict = "UNVERIFIED"
            verdict_label = "🟡 UNVERIFIED"
            reason = (
                "No matching articles found on the web, and the ML model itself isn't "
                f"confident either way ({ml_label.title()}, {ml_confidence:.1f}% confidence). "
                "This does not prove the claim is false — it may simply be too recent, "
                "too niche, or not yet covered."
            )

    return VerificationResult(
        query=claim,
        evidence=evidence,
        verdict=verdict,
        verdict_label=verdict_label,
        reason=reason,
        matched_sources=matched_sources,
    )
