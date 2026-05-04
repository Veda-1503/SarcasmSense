from preprocessing import preprocess
from clause_detection import detect_clauses
from domain_classifier import classify_domain
from sarcasm_classifier import detect_sarcasm
from sentiment_classifier import analyze_sentiment, invert_sentiment
from config import BASE_MODELS, SARCASM_THRESHOLD


def run_pipeline(text: str, domain: str = None) -> dict:
    preprocessed = preprocess(text)
    cleaned = preprocessed["cleaned"]
    sentences = preprocessed["sentences"]

    if not domain:
        domain = classify_domain(cleaned)
    domain = domain.lower().strip()

    clause_results = detect_clauses(sentences)

    sentence_results = []
    overall_sarcasm = False
    overall_sarcasm_prob = 0.0

    for i, cr in enumerate(clause_results):
        sent_text = cr["sentence"]
        clause_type = cr["clause_type"]

        sarcasm_result = detect_sarcasm(sent_text, domain)
        sentiment_result = analyze_sentiment(sent_text)

        is_sarcastic = sarcasm_result["is_sarcasm"]
        if is_sarcastic and sarcasm_result["sarcasm_prob"] > overall_sarcasm_prob:
            overall_sarcasm = True
            overall_sarcasm_prob = sarcasm_result["sarcasm_prob"]

        base_sentiment = sentiment_result["label"]
        base_confidence = sentiment_result["confidence"]

        if is_sarcastic:
            if sarcasm_result["level"] == "High":
                final_sentiment = invert_sentiment(base_sentiment)
                explanation = f"Sarcasm (High) detected → sentiment inverted."
            elif sarcasm_result["level"] == "Medium" and base_sentiment == "positive":
                final_sentiment = "negative"
                explanation = f"Sarcasm (Medium) detected → adjusted to negative."
            else:
                final_sentiment = base_sentiment
                explanation = f"Sarcasm ({sarcasm_result['level']}) detected → adjusted sentiment."
        else:
            final_sentiment = base_sentiment
            explanation = "No sarcasm detected → sentiment retained."

        sentence_results.append({
            "sentence": sent_text,
            "clause": clause_type,
            "sarcasm": sarcasm_result,
            "sentiment": {
                "base_label": base_sentiment,
                "base_confidence": base_confidence,
                "final_label": final_sentiment,
            },
            "explanation": explanation,
        })

    if sentence_results:
        label_counts = {}
        for sr in sentence_results:
            lbl = sr["sentiment"]["final_label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        score_map = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        for sr in sentence_results:
            lbl = sr["sentiment"]["final_label"]
            conf = sr["sentiment"]["base_confidence"]

            weight = 1.2 if sr["clause"] in ["Complex", "Compound-Complex"] else 1.0

            if lbl == "negative":
                score_map[lbl] += conf * weight * 1.25
            else:
                score_map[lbl] += conf * weight

        overall_final_sentiment = max(score_map, key=score_map.get)

        if (score_map["negative"] > 0.9 * score_map[overall_final_sentiment] and overall_final_sentiment == "positive"):
            overall_final_sentiment = "negative"
            
        overall_confidence = score_map[overall_final_sentiment] / max(1, len(sentence_results))
    
    else:
        overall_final_sentiment = "neutral"
        overall_confidence = 0.5

    model_used = BASE_MODELS.get(domain, "distilbert-base-uncased")

    return {
        "text": text,
        "domain": domain,
        "sentence_results": sentence_results,
        "overall_sarcasm": overall_sarcasm,
        "overall_sarcasm_prob": round(overall_sarcasm_prob, 4),
        "final_sentiment": overall_final_sentiment,
        "final_confidence": round(overall_confidence, 4),
        "model_used": model_used,
    }