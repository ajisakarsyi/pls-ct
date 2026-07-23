/* ================================================================
   CSIPBLLM — Frontend Script
   - Normalisasi LaTeX sebelum MathJax render
   - Setiap pertanyaan lanjutan membuat kartu baru
   - Alur benar: tutup sesi tanpa pertanyaan lanjutan
   - Alur salah: kartu followup baru + input tetap aktif
   ================================================================ */

document.addEventListener("DOMContentLoaded", () => {

  // ===========================================================
  // 1. Load marked.js
  // ===========================================================
  const loadMarked = () =>
    new Promise((resolve, reject) => {
      if (window.marked) { resolve(window.marked); return; }
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
      s.onload  = () => resolve(window.marked);
      s.onerror = () => reject(new Error("Gagal memuat marked.js"));
      document.head.appendChild(s);
    });

  loadMarked().catch((err) => console.error("marked.js error:", err));

  // ===========================================================
  // 2. Elemen DOM
  // ===========================================================
  const sendBtn          = document.getElementById("sendBtn");
  const evalBtn          = document.getElementById("evalBtn");
  const chatBox          = document.getElementById("chatBox");
  const userAnswer       = document.getElementById("userAnswer");
  const evalResult       = document.getElementById("evalResult");
  const answerSection    = document.getElementById("answerSection");
  const historySection   = document.getElementById("historySection");
  const questionInput    = document.getElementById("question");
  const downloadTxt      = document.getElementById("downloadTxt");
  const downloadJson     = document.getElementById("downloadJson");
  const followupSection  = document.getElementById("followupSection");
  const followupContainer= document.getElementById("followupContainer");
  const historyButtons   = document.getElementById("historyButtons");
  const rlBadge          = document.getElementById("rlBadge");
  const rlLtText         = document.getElementById("rlLtText");
  const rlPhaseText      = document.getElementById("rlPhaseText");
  const rlEpsText        = document.getElementById("rlEpsText");

  // ===========================================================
  // 2b. Dropdown Profil Kognitif (sederhana)
  // "Otomatis" → null → RL Agent yang memilih
  // Pilihan lain → kode profil dikirim langsung ke /chat
  // ===========================================================
  const cognitiveSelect = document.getElementById("cognitiveSelect");

  // ── Populate dropdown: opsi otomatis + 48 kombinasi profil kognitif ──
  // Kode dihasilkan persis sama dengan VALID_COGNITIVE_TYPES
  // di app/core/cognitive.py:
  //   f"{n}{pt}{ag}{ir}" for n in ["1".."6"]
  //                       for pt in ["P","T"]
  //                       for ag in ["A","G"]
  //                       for ir in ["I","R"]
  const LEVEL_DESC = {
    "1": "Lvl 1 – Remember",
    "2": "Lvl 2 – Understand",
    "3": "Lvl 3 – Apply",
    "4": "Lvl 4 – Analyze",
    "5": "Lvl 5 – Evaluate",
    "6": "Lvl 6 – Create",
  };
  const PT_MAP = { P: "Pragmatis", T: "Teoritis"  };
  const AG_MAP = { A: "Analitis",  G: "Global"    };
  const IR_MAP = { I: "Intuitif",  R: "Reflektif" };

  if (cognitiveSelect) {
    const autoOpt = document.createElement("option");
    autoOpt.value = "";
    autoOpt.textContent = "Otomatis (RL Agent)";
    cognitiveSelect.appendChild(autoOpt);

    ["1","2","3","4","5","6"].forEach(lvl => {
      ["P","T"].forEach(pt => {
        ["A","G"].forEach(ag => {
          ["I","R"].forEach(ir => {
            const code  = `${lvl}${pt}${ag}${ir}`;
            const label = `${code} — ${LEVEL_DESC[lvl]}, ${PT_MAP[pt]}, ${AG_MAP[ag]}, ${IR_MAP[ir]}`;
            const opt   = document.createElement("option");
            opt.value       = code;
            opt.textContent = label;
            cognitiveSelect.appendChild(opt);
          });
        });
      });
    });
    cognitiveSelect.value = ""; // default: otomatis
  }

  // Helper: cognitive yang akan dikirim ke /chat
  // "" (Otomatis) → null → RL Agent pilih sendiri
  // Selain itu     → kode profil terpilih
  const getActiveCognitive = () => {
    if (cognitiveSelect && cognitiveSelect.value) {
      return cognitiveSelect.value;
    }
    return null;
  };

  // ===========================================================
  // 3. State global
  // ===========================================================
  let correctAnswer  = "";   // penjelasan tutor (konteks)
  let activeQuestion = "";   // pertanyaan spesifik yang sedang dijawab mahasiswa
  let wrongAttempts  = 0;
  let followupCount  = 0;

  // ===========================================================
  // 4. Normalisasi LaTeX sisi klien
  // Tangkap kasus yang lolos dari server: [ \cmd... ] dan $...$
  // ===========================================================
  const normalizeLatex = (text) => {
    if (!text) return text;

    // Lindungi blok kode dengan placeholder yang aman (tanpa null byte)
    const codeStash = [];
    text = text.replace(/```[\s\S]*?```|`[^`]+`/g, (m) => {
      codeStash.push(m);
      return `CODEPH${codeStash.length - 1}PH`;
    });

    // $$ ... $$ → \[ ... \]
    text = text.replace(/\$\$(.+?)\$\$/gs, (_, m) => `\\[${m}\\]`);

    // $ ... $ (inline) → \( ... \)
    text = text.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_, m) => `\\(${m}\\)`);

    // LLM sering menulis \[\int... \\] dengan \\ ekstra — bersihkan
    // Pola: \[ ... \\ ] → \[ ... ]  (dalam konteks closing \])
    text = text.replace(/(\\\[[\s\S]*?)\\\\(\s*\\\])/g, (_, a, b) => a + b);

    // [ \latexCmd... ] atau [ \latexCmd... \\] → \[ ... \]
    const latexPattern = /\[\s*(\\(?:frac|int|sum|prod|lim|sqrt|left|right|begin|end|alpha|beta|gamma|delta|theta|lambda|mu|sigma|omega|pi|infty|partial|nabla|cdot|times|leq|geq|neq|approx|in|forall|exists|mathbb|mathbf|mathrm|text|overline|hat|vec|bar)[\s\S]*?)\s*(?:\\\\)?\]/g;
    text = text.replace(latexPattern, (_, m) => `\\[${m}\\]`);

    // Kembalikan blok kode
    codeStash.forEach((block, i) => {
      text = text.replace(`CODEPH${i}PH`, block);
    });

    return text;
  };

  // ===========================================================
  // 5. Utilitas render
  // ===========================================================
  const escapeHtml = (text) =>
    (text || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  // Render markdown sambil melindungi blok LaTeX dari marked.js
  const renderMarkdown = (raw) => {
    if (!raw) return "";

    // 1. Normalisasi notasi LaTeX
    let text = normalizeLatex(raw);

    // 2. Stash semua LaTeX dengan placeholder yang aman (hanya huruf/angka/underscore)
    //    null bytes dan karakter aneh rusak di dalam marked.js
    const mathStash = [];
    const stash = (m) => {
      const id = mathStash.length;
      mathStash.push(m);
      return `ZZZMATH${id}ZZZ`;
    };

    // Blok display \[ ... \] — tangkap dulu yang lebih panjang
    text = text.replace(/\\\[[\s\S]*?\\\]/g, stash);
    // Inline \( ... \)
    text = text.replace(/\\\([\s\S]*?\\\)/g, stash);

    // 3. marked.js hanya pada teks non-math
    if (window.marked) {
      text = window.marked.parse(text);
    } else {
      // fallback: bungkus dalam <p> sederhana
      text = `<p>${escapeHtml(text)}</p>`;
    }

    // 4. Kembalikan LaTeX verbatim — jangan di-escape
    mathStash.forEach((math, i) => {
      // marked mungkin sudah wrap placeholder dalam <p> atau <code> — cari dan ganti
      text = text.replace(`ZZZMATH${i}ZZZ`, math);
    });

    return text;
  };

  // Minta MathJax render ulang pada elemen tertentu
  const rerenderMath = (el) => {
    if (!el) return;
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([el]).catch((err) =>
        console.error("MathJax typesetPromise error:", err)
      );
    }
  };

  const getTrimmedValue = (el) =>
    el && typeof el.value === "string" ? el.value.trim() : "";

  const getRawValue = (el, fallback = "") =>
    el ? (el.value !== "" ? el.value : fallback) : fallback;

  const setBusy = (btn, busy) => { if (btn) btn.disabled = busy; };

  const triggerDownload = (blob, filename) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    a.style.display = "none";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  // ===========================================================
  // 6. Tambah bubble ke chatBox
  // ===========================================================
  const appendBubble = (role, html) => {
    if (!chatBox) return null;
    const placeholder = chatBox.querySelector(".placeholder");
    if (placeholder) placeholder.remove();

    const div = document.createElement("div");
    div.className = `bubble bubble-${role}`;
    div.innerHTML = html;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    rerenderMath(div);
    return div;
  };

  // ===========================================================
  // 7. Tambah KARTU BARU untuk setiap pertanyaan lanjutan
  // ===========================================================
  const appendFollowupCard = (text) => {
    if (!text || !text.trim()) return;
    if (!followupContainer || !followupSection) return;

    followupCount++;
    followupSection.style.display = "block";

    const card = document.createElement("div");
    card.className = "followup-card";

    const badge = document.createElement("span");
    badge.className = "followup-badge";
    badge.textContent = `Pertanyaan ${followupCount}`;

    const body = document.createElement("p");
    body.className = "followup-body math-render";

    card.appendChild(badge);
    card.appendChild(body);
    followupContainer.appendChild(card);

    body.innerHTML = renderMarkdown(text);
    rerenderMath(card);

    // Pertanyaan ini menjadi acuan evaluasi berikutnya
    activeQuestion = text;

    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  const clearFollowups = () => {
    followupCount  = 0;
    activeQuestion = "";
    if (followupContainer) followupContainer.innerHTML = "";
    if (followupSection)   followupSection.style.display = "none";
  };

  // ===========================================================
  // 8. Update RL status badge
  // ===========================================================
  const updateRlBadge = (data) => {
    if (!rlBadge || !data) return;
    const lt    = data.rl_selected_lt  || data.learning_type || "—";
    const phase = data.rl_phase?.phase || "";
    const eps   = data.rl_epsilon      != null ? `ε=${data.rl_epsilon.toFixed(3)}` : "";
    if (rlLtText)   rlLtText.textContent   = lt;
    if (rlPhaseText) rlPhaseText.textContent = phase ? `[${phase}]` : "";
    if (rlEpsText)   rlEpsText.textContent  = eps;
    rlBadge.style.display = "flex";
  };

  // ===========================================================
  // 8b. KONDISI A/B (revisi pasca-sidang item 1)
  // ===========================================================
  const cognitiveRow = document.getElementById("cognitiveRow");

  const getActiveMode = () => {
    const checked = document.querySelector('input[name="modeRadio"]:checked');
    return checked ? checked.value : "A";
  };

  // Mode B: sembunyikan elemen alur tutor yang tidak relevan
  // (profil kognitif, RL badge, followup, evaluasi jawaban)
  const applyModeVisibility = () => {
    const isB = getActiveMode() === "B";
    if (cognitiveRow)    cognitiveRow.style.display    = isB ? "none" : "flex";
    if (rlBadge && isB)  rlBadge.style.display         = "none";
    if (isB) {
      clearFollowups();
      if (answerSection)  answerSection.style.display  = "none";
      if (historySection) historySection.style.display = "none";
    }
  };
  document.querySelectorAll('input[name="modeRadio"]').forEach((r) =>
    r.addEventListener("change", applyModeVisibility)
  );

  // ===========================================================
  // 8c. PANEL DETAIL TRANSPARANSI (revisi item 2 & 3)
  //     Ditempel di bawah setiap jawaban: chunk yang dipakai
  //     (file + topik + skor + ambang), metrik live dengan rumus,
  //     bukti no_rag_proof (mode B), dan prompt yang dikirim.
  // ===========================================================
  const fmtNum = (v) => (v == null ? "—" : Number(v).toFixed(4));

  const buildTransparencyHtml = (data) => {
    const isB   = data.mode === "B";
    const parts = [];

    // — status kondisi —
    parts.push(
      `<p><span class="mode-chip ${isB ? "b" : "a"}">KONDISI ${data.mode}</span>` +
      (isB
        ? `RAG <span class="no">TIDAK DIPAKAI</span> — prompt buta Lampiran 4.`
        : `RAG <span class="ok">DIPAKAI</span> — jawaban dibangun dari chunk di bawah.`) +
      `</p>`
    );

    // — bukti Kondisi B —
    if (isB && data.no_rag_proof) {
      const p = data.no_rag_proof;
      parts.push(
        `<p><b>Bukti LLM murni (no_rag_proof):</b></p>` +
        `<table><tr><th>guard aktif</th><th>upaya retrieval diblokir</th>` +
        `<th>upaya embedding diblokir</th></tr>` +
        `<tr><td class="ok">${p.guard_enforced ? "✓ ya" : "✗"}</td>` +
        `<td>${p.retrieval_calls_blocked}</td>` +
        `<td>${p.embedding_calls_blocked}</td></tr></table>` +
        `<p class="na-line">Nilai 0 = tidak ada satu pun jalur kode yang mencoba ` +
        `menyentuh RAG selama jawaban ini dibuat. Guard tetap akan melempar ` +
        `error bila ada yang mencoba.</p>`
      );
    }

    // — chunk yang dipakai (mode A) —
    if (!isB && Array.isArray(data.retrieved) && data.retrieved.length) {
      const lm = data.live_metrics || {};
      const th  = lm.theta   != null ? lm.theta   : 0.25;
      const thc = lm.theta_c != null ? lm.theta_c : 0.35;
      const rows = data.retrieved.map((c) =>
        `<tr><td>${c.rank}</td><td>${escapeHtml(c.source || "—")}</td>` +
        `<td>${escapeHtml(c.topic || "—")}</td><td>${fmtNum(c.score)}</td>` +
        `<td>${c.score >= th  ? '<span class="ok">✓</span>' : '<span class="no">✗</span>'}</td>` +
        `<td>${c.score >= thc ? '<span class="ok">✓</span>' : '<span class="no">✗</span>'}</td></tr>`
      ).join("");
      parts.push(
        `<p><b>Chunk materi yang dipakai menjawab (Top-K):</b></p>` +
        `<table><tr><th>#</th><th>file sumber</th><th>topik</th><th>skor cosine</th>` +
        `<th>≥ θ=${th}</th><th>≥ θc=${thc}</th></tr>${rows}</table>`
      );
    }

    // — metrik live —
    const lm = data.live_metrics;
    if (lm) {
      const calcs = [];
      ["precision_detail", "coverage_detail", "mean_sim_detail",
       "source_diversity_detail"].forEach((k) => {
        if (lm[k] && lm[k].calc_str)
          calcs.push(`<div class="calc-line">∴ ${escapeHtml(lm[k].calc_str)}</div>`);
      });
      if (lm.faithfulness_live && lm.faithfulness_live.calc_str)
        calcs.push(`<div class="calc-line">∴ ${escapeHtml(lm.faithfulness_live.calc_str)}</div>`);
      if (calcs.length)
        parts.push(`<p><b>Metrik live jawaban ini (rumus + substitusi):</b></p>` + calcs.join(""));

      const lex = [];
      if (lm.uncertainty)
        lex.push(`<div class="calc-line">Uncertainty = ${lm.uncertainty.value} — ` +
                 `${escapeHtml(lm.uncertainty.keterangan || "")}</div>`);
      if (lm.contradiction)
        lex.push(`<div class="calc-line">Contradiction = ${lm.contradiction.value} — ` +
                 `${escapeHtml(lm.contradiction.keterangan || "")}</div>`);
      const matches = []
        .concat((lm.uncertainty?.matches || []).map((m) => ["Uncertainty", m]))
        .concat((lm.contradiction?.matches || []).map((m) => ["Contradiction", m]));
      if (matches.length) {
        const mrows = matches.map(([jenis, m]) =>
          `<tr><td>${jenis}</td><td>${escapeHtml(m.phrase)}</td>` +
          `<td>${(m.sentence_index ?? 0) + 1}</td><td>${m.char_start}–${m.char_end}</td>` +
          `<td>${escapeHtml(m.sentence || "")}</td></tr>`).join("");
        lex.push(`<table><tr><th>jenis</th><th>frasa</th><th>kalimat ke-</th>` +
                 `<th>posisi</th><th>kutipan</th></tr>${mrows}</table>`);
      }
      if (lex.length) parts.push(lex.join(""));

      if (lm.not_computed && Object.keys(lm.not_computed).length) {
        const na = Object.entries(lm.not_computed).map(([k, v]) =>
          `<div class="na-line">• ${escapeHtml(k)}: ${escapeHtml(v)}</div>`).join("");
        parts.push(`<p><b>Tidak dihitung live (dengan alasan):</b></p>` + na);
      }
    }

    // — prompt yang dikirim —
    if (data.prompt_sent) {
      parts.push(
        `<p><b>Prompt persis yang dikirim ke LLM${isB ? " (perhatikan: tidak ada konteks/profil/instruksi)" : ""}:</b></p>` +
        `<pre class="prompt-echo">${escapeHtml(data.prompt_sent)}</pre>`
      );
    }

    return `<details class="transparency"><summary>🔍 Detail Transparansi — ` +
           `Kondisi ${data.mode}${isB ? " (bukti LLM murni)" : " (chunk, skor & metrik live)"}` +
           `</summary><div class="tr-body">${parts.join("")}</div></details>`;
  };

  const appendTransparency = (data) => {
    if (!chatBox || !data || !data.mode) return;
    const div = document.createElement("div");
    div.innerHTML = buildTransparencyHtml(data);
    chatBox.appendChild(div.firstChild);
    chatBox.scrollTop = chatBox.scrollHeight;
  };

  // ===========================================================
  // 9. Kirim pertanyaan ke /chat
  // ===========================================================
  const sendQuestion = async () => {
    const message = getTrimmedValue(questionInput);

    if (!message) { alert("Tulis pertanyaan terlebih dahulu!"); return; }

    appendBubble("user", `<b>Kamu:</b> ${escapeHtml(message)}`);
    const loading = appendBubble("status", "<i>Memproses jawaban...</i>");
    setBusy(sendBtn, true);
    wrongAttempts  = 0;
    activeQuestion = "";   // reset — pertanyaan awal belum ada followup
    clearFollowups();
    if (historyButtons) historyButtons.style.display = "none";
    evalResult.className = "eval-result";
    evalResult.innerHTML = "";
    if (userAnswer) { userAnswer.disabled = false; userAnswer.value = ""; }
    if (evalBtn)    evalBtn.disabled = false;

    try {
      // Kondisi eksperimen (revisi pasca-sidang): A = RAG, B = LLM murni
      const mode        = getActiveMode();
      // Demo mode: kirim cognitive jika dipilih manual (override RL)
      // Jika null → RL Agent yang memilih (perilaku normal). Mode B:
      // profil tidak dipakai sama sekali.
      const demoCognitive = getActiveCognitive();
      const chatPayload   = { message, session_id: "default", mode };
      if (mode === "A" && demoCognitive) chatPayload.cognitive = demoCognitive;

      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(chatPayload),
      });

      const data = await res.json();
      if (loading) loading.remove();

      if (data.error) {
        appendBubble("bot", `❌ ${escapeHtml(data.error)}`);
        return;
      }

      const isB = data.mode === "B";
      if (isB) {
        if (rlBadge) rlBadge.style.display = "none";
      } else {
        updateRlBadge(data);
      }

      const chip  = `<span class="mode-chip ${isB ? "b" : "a"}">KONDISI ${data.mode || "A"}</span>`;
      const who   = isB ? "LLM murni (tanpa RAG)" : `Tutor (${escapeHtml(data.cognitive || "—")})`;
      appendBubble(
        "bot",
        `${chip}<b>${who}:</b><br>${renderMarkdown(data.reply || "")}`
      );

      // Panel transparansi: chunk+topik+skor, metrik live, no_rag_proof, prompt
      appendTransparency(data);

      if (!isB && data.followup_question) {
        appendFollowupCard(data.followup_question);
      }

      correctAnswer = data.reply || "";
      if (!isB) {
        if (answerSection)  answerSection.style.display  = "block";
        if (historySection) historySection.style.display = "block";
      }

    } catch (err) {
      if (loading) loading.remove();
      appendBubble("bot", `❌ Gagal memproses: ${escapeHtml(String(err))}`);
    } finally {
      setBusy(sendBtn, false);
    }
  };

  if (sendBtn) sendBtn.addEventListener("click", sendQuestion);
  if (questionInput) {
    questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
    });
  }

  // ===========================================================
  // 10. Evaluasi jawaban via /evaluate
  // ===========================================================
  const evaluateAnswer = async () => {
    const answer = getTrimmedValue(userAnswer);

    if (!answer)        { alert("Tulis jawabanmu dulu!"); return; }
    if (!correctAnswer) { alert("Belum ada jawaban referensi dari tutor. Kirim pertanyaan dulu."); return; }

    setBusy(evalBtn, true);
    evalResult.className = "eval-result";
    evalResult.textContent = "Menilai jawaban...";

    try {
      const res = await fetch("/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer,
          correct_answer:  correctAnswer,
          active_question: activeQuestion,
          wrong_count:     wrongAttempts,
          session_id:      "default",
        }),
      });

      const data = await res.json();
      evalResult.classList.remove("correct", "incorrect");

      // Update RL badge from evaluate response
      if (data.rl) updateRlBadge(data.rl);

      if (data.is_correct) {
        // ── BENAR ─────────────────────────────────────────────
        evalResult.classList.add("correct");
        if (historyButtons) historyButtons.style.display = "block";

        // Hapus semua followup — sesi selesai
        clearFollowups();

        evalResult.innerHTML = `
          <p><b>✅ Jawabanmu BENAR!</b></p>
          <div class="feedback-body math-render">${renderMarkdown(data.feedback || "")}</div>
        `;

        // Kunci input — sesi ini selesai
        if (userAnswer) userAnswer.disabled = true;
        if (evalBtn)    evalBtn.disabled    = true;

      } else {
        // ── SALAH ─────────────────────────────────────────────
        wrongAttempts += 1;
        evalResult.classList.add("incorrect");

        evalResult.innerHTML = `
          <p><b>❌ Jawabanmu belum tepat.</b></p>
          <span class="hint-badge">${escapeHtml(data.hint_level || "Evaluasi Awal")}</span>
          <div class="feedback-body math-render">${renderMarkdown(data.feedback || "")}</div>
        `;

        // Tambahkan kartu followup BARU (bukan timpa yang lama)
        if (data.followup_question) {
          appendFollowupCard(data.followup_question);
        }

        // Kosongkan input agar mahasiswa bisa coba lagi
        if (userAnswer) { userAnswer.value = ""; userAnswer.focus(); }
      }

      rerenderMath(evalResult);

    } catch (err) {
      evalResult.className   = "eval-result";
      evalResult.textContent = `❌ Gagal evaluasi: ${String(err)}`;
    } finally {
      setBusy(evalBtn, false);
    }
  };

  if (evalBtn) evalBtn.addEventListener("click", evaluateAnswer);
  if (userAnswer) {
    userAnswer.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); evaluateAnswer(); }
    });
  }

  // ===========================================================
  // 11. Download riwayat
  // ===========================================================
  if (downloadTxt) {
    downloadTxt.addEventListener("click", async () => {
      try {
        const res  = await fetch("/history?format=txt");
        const data = await res.json();
        triggerDownload(new Blob([data.data || ""], { type: "text/plain" }), "riwayat.txt");
      } catch (err) {
        appendBubble("bot", `❌ Gagal mengunduh TXT: ${escapeHtml(String(err))}`);
      }
    });
  }

  if (downloadJson) {
    downloadJson.addEventListener("click", async () => {
      try {
        const res  = await fetch("/history?format=json");
        const data = await res.json();
        triggerDownload(
          new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
          "riwayat.json"
        );
      } catch (err) {
        appendBubble("bot", `❌ Gagal mengunduh JSON: ${escapeHtml(String(err))}`);
      }
    });
  }
});