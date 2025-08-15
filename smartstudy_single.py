# smartstudy_single.py
# SmartStudy — 2025 Professional UI (single-file FastAPI app)
# Fix: ensure buttons reliably bind by running all DOM code after DOMContentLoaded.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- Naive Summarizer (pure Python) ----------
import re
from collections import Counter, defaultdict

STOPWORDS = set("""
a an the and is are was were in on of for to with by that this it as at from be or
about into over after before under between during through up down out off above below
can could should would may might shall will just not no nor so such than then too very
""".split())

def _tokenize_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in sentences if s.strip()]

def _tokenize_words(sentence: str):
    words = re.findall(r'\w+', sentence.lower())
    return [w for w in words if w not in STOPWORDS]

def summarize(text: str, max_sentences: int = 3) -> str:
    sents = _tokenize_sentences(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)
    freq = Counter()
    for s in sents:
        for w in _tokenize_words(s):
            freq[w] += 1
    scores = defaultdict(int)
    for i, s in enumerate(sents):
        for w in _tokenize_words(s):
            scores[i] += freq[w]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    idx = sorted(i for i, _ in ranked[:max_sentences])
    return " ".join(sents[i] for i in idx)

# ---------- Flashcard Generator (rule-based) ----------
from typing import List, Tuple

def _sent_split(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in sents if s.strip()]

def _qa_from_sentence(sentence: str) -> Tuple[str, str]:
    s = sentence.strip()

    m = re.search(r'^(.*?)\s+(is|are|was|were)\s+(.*)', s, re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        rest = m.group(3).strip()
        return (f"What is {subject}?", rest.rstrip(".!?"))

    m2 = re.search(r'^(.*?)\s+has\s+(.*)', s, re.IGNORECASE)
    if m2:
        subject = m2.group(1).strip()
        obj = m2.group(2).strip()
        return (f"What does {subject} have?", obj.rstrip(".!?"))

    words = s.split()
    if len(words) > 6:
        return (f"What does this say about: {' '.join(words[:6])}…", s)
    return (f"Question: {s[:40]}...", s)

def generate_flashcards(text: str, max_cards: int = 10) -> List[Tuple[str, str]]:
    cards = []
    for sent in _sent_split(text):
        q, a = _qa_from_sentence(sent)
        cards.append((q, a))
        if len(cards) >= max_cards:
            break
    return cards

# ---------- OCR Utilities ----------
from PIL import Image
import io
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

def available_languages() -> List[str]:
    if not TESS_AVAILABLE:
        return ["eng"]
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return ["eng"]

def ocr_image_bytes(file_bytes: bytes, lang: str = "eng") -> str:
    if not TESS_AVAILABLE:
        raise RuntimeError("Tesseract not available. Install system package. (ગુજરાતી: Tesseract ઇન્સ્ટોલ કરો.)")

    if len(file_bytes) > 12 * 1024 * 1024:
        raise RuntimeError("File too large (>12MB). Please use a smaller image.")

    image = Image.open(io.BytesIO(file_bytes))

    requested = [l.strip() for l in (lang or "eng").split("+") if l.strip()]
    installed = set(available_languages())
    missing = [l for l in requested if l not in installed]
    if missing:
        raise RuntimeError(
            "Missing language data: "
            + ", ".join(missing)
            + ". Install the missing .traineddata files. "
            + "(ગુજરાતી: જરૂરી ભાષાની .traineddata ફાઇલ ઇન્સ્ટોલ કરો.)"
        )
    lang_arg = "+".join(requested) if requested else "eng"
    return pytesseract.image_to_string(image, lang=lang_arg)

# ---------- FastAPI App ----------
app = FastAPI(title="SmartStudy — Single File")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- UI (fixed JS: all bindings run after DOMContentLoaded) ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>📘 StudySnap — SmartStudy</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="https://upload.wikimedia.org/wikipedia/commons/7/73/Iconic_image_of_a_book.png" />
<style>
/* (CSS identical to previous professional UI; omitted here for brevity in message) */
:root{ --accent:#2563eb; --accent-2:#14b8a6; --radius:16px; --ring:#93c5fd; }
body{ font-family:Inter,system-ui; margin:0; background:linear-gradient(135deg,#eef4ff,#fffaf3); }
.header{ position:sticky; top:0; background:rgba(255,255,255,.76); padding:12px 20px; }
.header-inner{ max-width:1200px; margin:0 auto; display:flex; justify-content:space-between; align-items:center; }
.brand{ font-weight:800; display:flex; gap:10px; align-items:center; }
.container{ max-width:1200px; margin:18px auto; padding:0 20px; }
.grid{ display:grid; grid-template-columns:1.15fr .85fr; gap:20px; }
@media(max-width:980px){ .grid{ grid-template-columns:1fr } }
.card{ background:rgba(255,255,255,.78); border-radius:16px; padding:18px; box-shadow:0 12px 40px rgba(2,6,23,0.12); }
.drop{ border:2px dashed rgba(100,116,139,.5); padding:22px; border-radius:12px; text-align:center; }
.controls{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:10px; }
select, textarea{ width:100%; padding:12px; border-radius:10px; border:1px solid rgba(148,163,184,.4); }
textarea{ min-height:150px; font-family:monospace; }
.btn{ padding:10px 14px; border-radius:10px; border:none; background:linear-gradient(90deg,var(--accent),var(--accent-2)); color:#fff; cursor:pointer; font-weight:700; }
.btn.secondary{ background:transparent; border:1px solid rgba(148,163,184,.4); color:#222; }
.output{ background:rgba(240,247,255,.9); padding:12px; border-radius:10px; margin-top:12px; font-family:monospace; white-space:pre-wrap; }
.kbd{ padding:2px 6px; border-radius:6px; background:#f3f4f6; border:1px solid rgba(148,163,184,.4); font-size:.78rem; }
.toast{ position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:rgba(2,6,23,.9); color:#fff; padding:10px 14px; border-radius:10px; display:none; }
.toast.show{ display:block; }
</style>
</head>
<body>
  <div class="header">
    <div class="header-inner">
      <div class="brand"><div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#fff;display:grid;place-items:center">📘</div><div>StudySnap — SmartStudy</div></div>
      <div class="actions"><span class="kbd">Ctrl/⌘+Enter</span> <span class="kbd">Alt+Enter</span> <button id="themeToggle" class="btn secondary">🌗</button> <button id="helpBtn" class="btn secondary">❓</button></div>
    </div>
  </div>

  <div class="container">
    <div style="margin-top:12px;" class="tabs"><div class="tab active">Capture</div><div class="tab">Compose</div><div class="tab">Review</div></div>

    <div class="grid">
      <section class="card">
        <h3>Capture & OCR</h3>
        <div class="drop" id="drop">
          <div><strong>Drag & drop</strong> an image (JPG/PNG) — or paste from clipboard.</div>
          <input id="file" type="file" accept="image/*" />
          <div class="controls">
            <label class="helper">OCR languages:</label>
            <select id="langs" multiple size="6" style="min-width:180px"></select>
            <button id="btn-ocr" class="btn">Run OCR</button>
            <button id="btn-clear" class="btn secondary">Clear</button>
            <label style="display:flex;align-items:center;gap:6px;"><input id="autoRun" type="checkbox" /> Auto-run summarize</label>
          </div>
          <div id="ocr" class="output" style="min-height:72px">OCR output will appear here.</div>
        </div>

        <h3 style="margin-top:16px">Notes</h3>
        <textarea id="text" placeholder="Paste your notes here..."></textarea>
        <div class="controls" style="margin-top:10px">
          <button id="btn-sum" class="btn">Summarize (3 sentences)</button>
          <button id="btn-fc" class="btn">Generate Flashcards</button>
          <button id="btn-sample" class="btn secondary">Insert Sample</button>
        </div>
      </section>

      <section class="card">
        <div style="display:flex;gap:8px;justify-content:flex-end;margin-bottom:8px">
          <button id="btn-copy-summary" class="btn secondary">Copy Summary</button>
          <button id="btn-dl-summary" class="btn secondary">Download .txt</button>
          <button id="btn-export-csv" class="btn secondary">Export Cards .csv</button>
        </div>

        <h3>Summary (સારાંશ)</h3>
        <div id="summary" class="output" style="min-height:110px"></div>

        <h3 style="margin-top:14px">Flashcards (ફ્લેશકાર્ડ)</h3>
        <div id="cardsWrap" class="output" style="padding:12px">
          <div id="cardsEmpty" class="helper">Your flashcards will appear here.</div>
          <div id="cardsGrid" class="fc-grid" style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px"></div>
        </div>
      </section>
    </div>
  </div>

  <div id="toast" class="toast"></div>

<script>
// All DOM selection, listeners, and functions are initialized after DOMContentLoaded
window.addEventListener('DOMContentLoaded', function () {

  // Elements (safe to query now)
  const themeToggle = document.getElementById('themeToggle');
  const helpBtn = document.getElementById('helpBtn');
  const elText = document.getElementById('text');
  const elSum = document.getElementById('summary');
  const elFile = document.getElementById('file');
  const elOCR = document.getElementById('ocr');
  const elLangs = document.getElementById('langs');
  const elCardsGrid = document.getElementById('cardsGrid');
  const elCardsEmpty = document.getElementById('cardsEmpty');
  const autoRun = document.getElementById('autoRun');
  const drop = document.getElementById('drop');
  const toastEl = document.getElementById('toast');

  // Buttons
  const btnSum = document.getElementById('btn-sum');
  const btnFc = document.getElementById('btn-fc');
  const btnOcr = document.getElementById('btn-ocr');
  const btnClear = document.getElementById('btn-clear');
  const btnSample = document.getElementById('btn-sample');
  const btnCopySummary = document.getElementById('btn-copy-summary');
  const btnDlSummary = document.getElementById('btn-dl-summary');
  const btnExportCsv = document.getElementById('btn-export-csv');

  // Defensive checks
  function $(el){ return document.getElementById(el); } // helper if needed

  // Theme: system preferred remembered
  (function initTheme(){
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if(saved === 'dark' || (!saved && prefersDark)) {
      document.documentElement.classList.add('dark');
    }
  })();
  if(themeToggle) themeToggle.addEventListener('click', () => {
    const el = document.documentElement;
    const nowDark = !el.classList.contains('dark');
    el.classList.toggle('dark', nowDark);
    localStorage.setItem('theme', nowDark ? 'dark':'light');
  });

  // Toast
  function toast(msg) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    setTimeout(()=> toastEl.classList.remove('show'), 1800);
  }

  // Autosave notes
  (function initAutosave(){
    try {
      const saved = localStorage.getItem('studysnap_notes');
      if(saved) elText.value = saved;
      elText.addEventListener('input', () => localStorage.setItem('studysnap_notes', elText.value));
    } catch(e) { /* ignore */ }
  })();

  // Load OCR languages
  async function loadLangs() {
    try {
      const res = await fetch('/ocr/langs');
      const data = await res.json();
      const saved = (localStorage.getItem('ocr_langs') || '').split('+').filter(Boolean);
      const preferred = saved.length ? saved : ['eng','guj'];
      elLangs.innerHTML = '';
      (data.languages || ['eng']).sort().forEach(code => {
        const opt = document.createElement('option');
        opt.value = code; opt.textContent = code;
        if (preferred.includes(code)) opt.selected = true;
        elLangs.appendChild(opt);
      });
    } catch (e) {
      elLangs.innerHTML = '';
      ['eng','guj','hin','ben','tam','tel','kan','mal','pan','ori','san'].forEach(code => {
        const opt = document.createElement('option');
        opt.value = code; opt.textContent = code;
        if (['eng','guj'].includes(code)) opt.selected = true;
        elLangs.appendChild(opt);
      });
    }
  }

  // Button loading helper
  function setLoading(btn, isLoading, doneLabel) {
    if (!btn) return;
    if (isLoading) {
      btn.dataset._label = btn.innerHTML;
      btn.innerHTML = '<span style="display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.6);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;margin-right:8px"></span>' + (btn.dataset._label || btn.textContent || 'Working');
      btn.disabled = true;
    } else {
      btn.innerHTML = doneLabel || btn.dataset._label || btn.textContent || 'Action';
      btn.disabled = false;
    }
  }

  // Summarize
  async function summarize() {
    if (!btnSum) return;
    setLoading(btnSum, true);
    if (elSum) elSum.textContent = "Working...";
    try {
      const res = await fetch('/summarize', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ text: elText.value || "", max_sentences: 3 })
      });
      const data = await res.json();
      if (elSum) elSum.textContent = data.summary || "(no summary)";
    } catch (e) {
      if (elSum) elSum.textContent = "Error: " + (e?.message || e);
    } finally {
      setLoading(btnSum, false);
    }
  }

  // Flashcards
  async function flashcards() {
    if (!btnFc) return;
    setLoading(btnFc, true);
    if (elCardsGrid) elCardsGrid.innerHTML = "";
    if (elCardsEmpty) elCardsEmpty.style.display = "block";
    try {
      const res = await fetch('/flashcards', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ text: elText.value || "" })
      });
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) {
        if (elCardsGrid) elCardsGrid.innerHTML = "";
        if (elCardsEmpty) elCardsEmpty.style.display = "block";
        return;
      }
      if (elCardsEmpty) elCardsEmpty.style.display = "none";
      elCardsGrid.innerHTML = "";
      data.forEach((c, idx) => {
        const card = document.createElement('div');
        card.className = 'fc';
        const q = (c.question || "").replace(/^Summarize:\s*/i, "Topic: ");
        card.innerHTML = "<div style='font-weight:700;margin-bottom:6px'>" + (idx+1) + ". " + q + "</div>"
                       + "<div><b>A:</b> " + (c.answer || "") + "</div>";
        elCardsGrid.appendChild(card);
      });
    } catch (e) {
      if (elCardsGrid) elCardsGrid.innerHTML = "<div class='helper'>Error: " + (e?.message || e) + "</div>";
    } finally {
      setLoading(btnFc, false);
    }
  }

  // Run OCR
  async function runOCR() {
    if (!btnOcr) return;
    const f = elFile && elFile.files && elFile.files[0];
    if (!f) { toast("Choose an image first"); return; }
    if (!f.type.startsWith('image/')) { toast("Please choose an image file"); return; }

    if (elOCR) elOCR.textContent = "Running OCR...";
    setLoading(btnOcr, true);

    const selected = Array.from(elLangs.selectedOptions || []).map(o => o.value);
    const lang = selected.length ? selected.join('+') : 'eng';
    localStorage.setItem('ocr_langs', lang);

    const form = new FormData();
    form.append('file', f);
    form.append('lang', lang);

    try {
      const res = await fetch('/ocr', { method: 'POST', body: form });
      const data = await res.json();
      if (data.error) {
        if (elOCR) elOCR.textContent = "Error: " + data.error + "\\nGujarati: જરૂરી ભાષા ફાઇલ ઇન્સ્ટોલ કરો.";
        toast("OCR error");
      } else {
        if (elOCR) elOCR.textContent = data.text || "(no text found)";
        if (elText) elText.value = data.text || "";
        toast("OCR complete");
        if (autoRun && autoRun.checked) {
          await summarize();
          await flashcards();
        }
      }
    } catch (e) {
      if (elOCR) elOCR.textContent = "Error: " + (e?.message || e);
    } finally {
      setLoading(btnOcr, false);
    }
  }

  // Drag & drop + paste handlers
  ['dragenter','dragover'].forEach(ev => {
    drop && drop.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); drop.classList.add('drag'); });
  });
  ['dragleave','drop'].forEach(ev => {
    drop && drop.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); drop.classList.remove('drag'); });
  });
  drop && drop.addEventListener('drop', e => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) { toast('Please drop an image file'); return; }
    elFile.files = e.dataTransfer.files;
    runOCR();
  });
  window.addEventListener('paste', async (event) => {
    const items = event.clipboardData?.items;
    if (!items) return;
    for (const it of items) {
      if (it.type.startsWith('image/')) {
        const blob = it.getAsFile();
        const dt = new DataTransfer();
        dt.items.add(blob);
        elFile.files = dt.files;
        await runOCR();
        break;
      }
    }
  });

  // Clear
  btnClear && btnClear.addEventListener('click', () => {
    if (elFile) elFile.value = "";
    if (elOCR) elOCR.textContent = "OCR output will appear here.";
    if (elText) elText.value = "";
    if (elSum) elSum.textContent = "";
    if (elCardsGrid) elCardsGrid.innerHTML = "";
    if (elCardsEmpty) elCardsEmpty.style.display = "block";
    toast("Cleared");
  });

  // Sample
  btnSample && btnSample.addEventListener('click', () => {
    const sample = [
      "Artificial Intelligence (AI) simulates human intelligence in machines.",
      "It includes Narrow AI for specific tasks and General AI for human-like versatility.",
      "Key applications: NLP, computer vision, robotics, predictive analytics.",
      "Benefits: automation, accuracy, and data-driven decisions.",
      "Risks: job displacement, ethical dilemmas, misuse.",
      "Future impact depends on responsible development and regulation."
    ].join(" ");
    if (elText) elText.value = sample;
    try { localStorage.setItem('studysnap_notes', sample); } catch(e){}
    toast("Sample text inserted");
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', async (e) => {
    const meta = e.ctrlKey || e.metaKey;
    if (meta && e.key === 'Enter') { e.preventDefault(); await summarize(); }
    if (e.altKey && e.key === 'Enter') { e.preventDefault(); await flashcards(); }
  });

  // Copy / Download / Export
  btnCopySummary && btnCopySummary.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(elSum.textContent || ""); toast("Summary copied"); }
    catch { toast("Copy failed"); }
  });
  btnDlSummary && btnDlSummary.addEventListener('click', () => {
    const blob = new Blob([elSum.textContent || ""], {type: 'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'summary.txt'; a.click();
    URL.revokeObjectURL(a.href);
  });
  btnExportCsv && btnExportCsv.addEventListener('click', () => {
    const rows = [['Question','Answer']];
    Array.from(elCardsGrid.children || []).forEach(div => {
      const strong = (div.querySelector('div:first-child')?.textContent) || "";
      const ans = (div.querySelector('div:nth-child(2)')?.textContent || "").replace(/^A:\s*/,'');
      const q = strong.replace(/^\d+\.\s*/, '');
      rows.push([q, ans]);
    });
    const csv = rows.map(r => r.map(x => '"' + (x||'').replace(/"/g,'""') + '"').join(',')).join('\\n');
    const blob = new Blob([csv], {type:'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'flashcards.csv'; a.click();
    URL.revokeObjectURL(a.href);
  });

  // Help
  helpBtn && helpBtn.addEventListener('click', () => {
    toast("Tip: Select OCR languages (ENG/GUJ), drag/drop or paste an image, then Run OCR. Use Ctrl/⌘+Enter to Summarize.");
  });

  // Wire primary actions
  btnSum && btnSum.addEventListener('click', summarize);
  btnFc && btnFc.addEventListener('click', flashcards);
  btnOcr && btnOcr.addEventListener('click', runOCR);

  // Init
  loadLangs();
  // small CSS spinner keyframes injection (since we used inline spinner)
  const style = document.createElement('style');
  style.textContent = "@keyframes spin{ to{ transform: rotate(360deg); } }";
  document.head.appendChild(style);
});
</script>
</body>
</html>
"""

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML

@app.get("/ocr/langs")
async def api_ocr_langs():
    return {"languages": available_languages()}

@app.post("/summarize")
async def api_summarize(payload: dict):
    text = (payload or {}).get("text", "")
    k = (payload or {}).get("max_sentences", 3)
    try:
        k = int(k)
    except Exception:
        k = 3
    return {"summary": summarize(text, k)}

@app.post("/flashcards")
async def api_flashcards(payload: dict):
    text = (payload or {}).get("text", "")
    cards = generate_flashcards(text, max_cards=10)
    return [{"question": q, "answer": a} for q, a in cards]

@app.post("/ocr")
async def api_ocr(file: UploadFile = File(...), lang: str = Form("eng")):
    try:
        blob = await file.read()
        text = ocr_image_bytes(blob, lang=lang)
        return {"text": text, "lang": lang}
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("smartstudy_single:app", host="0.0.0.0", port=8000, reload=True)
