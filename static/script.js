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
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: "default" }),
      });

      const data = await res.json();
      if (loading) loading.remove();

      if (data.error) {
        appendBubble("bot", `❌ ${escapeHtml(data.error)}`);
        return;
      }

      updateRlBadge(data);

      appendBubble(
        "bot",
        `<b>Tutor (${escapeHtml(data.cognitive || "—")}):</b><br>${renderMarkdown(data.reply || "")}`
      );

      if (data.followup_question) {
        appendFollowupCard(data.followup_question);
      }

      correctAnswer = data.reply || "";
      if (answerSection)  answerSection.style.display  = "block";
      if (historySection) historySection.style.display = "block";

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