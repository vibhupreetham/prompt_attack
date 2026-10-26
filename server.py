#!/usr/bin/env python3
"""
PRISM-Guard: Multi-Stage Prompt Injection Defense Server
Serves the interactive web UI and provides real-time ML inference endpoints
using trained XGBoost models and TF-IDF vectorizer.
"""

import os
import json
import time
import joblib
import numpy as np
from http.server import SimpleHTTPRequestHandler, HTTPServer
import urllib.parse

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global model holders
vectorizer = None
clf_binary = None
clf_multiclass = None
clf_severity = None
model_metadata = None

attack_categories = ['benign', 'direct_injection', 'jailbreak_persona', 'system_override']

def load_models():
    global vectorizer, clf_binary, clf_multiclass, clf_severity, model_metadata
    try:
        vec_path = os.path.join(BASE_DIR, "outputs/models/vectorizer.pkl")
        bin_path = os.path.join(BASE_DIR, "outputs/models/clf_binary.pkl")
        mc_path = os.path.join(BASE_DIR, "outputs/models/clf_multiclass.pkl")
        sev_path = os.path.join(BASE_DIR, "outputs/models/clf_severity.pkl")
        meta_path = os.path.join(BASE_DIR, "outputs/models/model_metadata.json")

        if os.path.exists(vec_path):
            vectorizer = joblib.load(vec_path)
            print("✓ Loaded TF-IDF vectorizer")
        if os.path.exists(bin_path):
            clf_binary = joblib.load(bin_path)
            print("✓ Loaded Stage 1 Binary XGBoost Model")
        if os.path.exists(mc_path):
            clf_multiclass = joblib.load(mc_path)
            print("✓ Loaded Stage 2 Multi-Class XGBoost Model")
        if os.path.exists(sev_path):
            clf_severity = joblib.load(sev_path)
            print("✓ Loaded Stage 3 Severity XGBoost Model")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                model_metadata = json.load(f)
            print("✓ Loaded model metadata")
    except Exception as e:
        print(f"Warning loading models: {e}")

def analyze_prompt(text):
    t0 = time.time()
    if not text or not text.strip():
        return {
            "error": "Empty prompt provided",
            "latency_ms": 0
        }
    
    text = text.strip()
    
    # Check if models are loaded
    if vectorizer is not None and clf_binary is not None:
        X_vec = vectorizer.transform([text])
        
        # Stage 1: Binary Classification
        prob_malicious = float(clf_binary.predict_proba(X_vec)[0, 1])
        is_malicious = bool(prob_malicious >= 0.5)
        
        # Stage 2: Multi-Class Attack Taxonomy
        if clf_multiclass is not None:
            mc_probs = clf_multiclass.predict_proba(X_vec)[0]
            mc_idx = int(np.argmax(mc_probs))
            predicted_category = attack_categories[mc_idx] if mc_idx < len(attack_categories) else "unknown"
            mc_prob = float(mc_probs[mc_idx])
            mc_breakdown = {attack_categories[i]: round(float(mc_probs[i]), 4) for i in range(len(attack_categories))}
        else:
            predicted_category = "direct_injection" if is_malicious else "benign"
            mc_prob = prob_malicious
            mc_breakdown = {}
            
        # Stage 3: Severity / Bypass Likelihood
        if is_malicious and clf_severity is not None:
            sev_prob = float(clf_severity.predict_proba(X_vec)[0, 1])
        elif is_malicious:
            # Fallback heuristic if severity model absent
            sev_prob = min(0.95, prob_malicious * 0.85)
        else:
            sev_prob = max(0.01, prob_malicious * 0.1)
            
    else:
        # Fallback heuristic engine if PKLs not loaded
        t_lower = text.lower()
        score = 0.05
        adversarial_tokens = ["ignore", "disregard", "system prompt", "dan", "jailbreak", "override", "developer mode", "hypothetical", "bypass", "unfiltered", "rules", "instructions"]
        for tok in adversarial_tokens:
            if tok in t_lower:
                score += 0.22
        prob_malicious = min(0.99, score)
        is_malicious = prob_malicious >= 0.5
        predicted_category = "jailbreak_persona" if "dan" in t_lower or "developer mode" in t_lower else ("system_override" if "override" in t_lower or "ignore" in t_lower else ("direct_injection" if is_malicious else "benign"))
        mc_prob = prob_malicious
        mc_breakdown = {}
        sev_prob = prob_malicious * 0.8 if is_malicious else 0.02

    # Determine Severity Tier
    if sev_prob >= 0.75:
        severity_tier = "Critical"
        severity_color = "#ef4444"
        action_recommendation = "Immediate Block & Quarantine; Log High-Priority SOC Incident; Blacklist Session"
    elif sev_prob >= 0.50:
        severity_tier = "High"
        severity_color = "#f97316"
        action_recommendation = "Filter & Neutralize; Apply Strict Secondary Delimiter Sandboxing; Flag for Human Audit"
    elif sev_prob >= 0.25:
        severity_tier = "Medium"
        severity_color = "#f59e0b"
        action_recommendation = "Enforce System Prompt Reinforcement; Sanitize Delimiters; Monitor Multi-Turn Context"
    else:
        severity_tier = "Low"
        severity_color = "#10b981"
        action_recommendation = "Safe to Process; Normal Guardrail Filtering Sufficient"

    # Token-level risk attribution
    tokens = text.split()
    annotated_tokens = []
    top_rules = model_metadata.get("top_rules", {}) if model_metadata else {}
    
    text_lower = text.lower()
    for tok in tokens:
        clean = tok.lower().strip(".,!?:;\"'()[]{}")
        weight = 0.0
        if clean in top_rules:
            weight = float(top_rules[clean])
        elif any(k in clean for k in ["dan", "bypass", "jailbreak", "ignore", "override", "system", "prompt", "instructions", "rules", "sensitive", "confidential"]):
            weight = 0.035
        
        annotated_tokens.append({
            "token": tok,
            "weight": round(weight, 4),
            "is_risk": bool(weight > 0.008)
        })

    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "text": text,
        "is_malicious": is_malicious,
        "injection_probability": round(prob_malicious, 4),
        "stage1_decision": "MALICIOUS (Injection Detected)" if is_malicious else "BENIGN (Safe Request)",
        "stage2_attack_type": predicted_category,
        "stage2_confidence": round(mc_prob, 4),
        "stage2_breakdown": mc_breakdown,
        "stage3_bypass_likelihood": round(sev_prob, 4),
        "stage3_severity_tier": severity_tier,
        "severity_color": severity_color,
        "action_recommendation": action_recommendation,
        "annotated_tokens": annotated_tokens,
        "latency_ms": latency_ms
    }

class PrismHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                prompt_text = data.get('text', '')
                result = analyze_prompt(prompt_text)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/metrics':
            metrics_path = os.path.join(BASE_DIR, "outputs/eval/metrics.json")
            if os.path.exists(metrics_path):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                with open(metrics_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            super().do_GET()

def run_server():
    load_models()
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, PrismHandler)
    print("="*65)
    print(f"  PRISM-Guard Live Server running at: http://localhost:{PORT}")
    print(f"  Open in your browser to test and present to your professor!")
    print(f"  API Endpoint: POST http://localhost:{PORT}/api/predict")
    print("="*65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
