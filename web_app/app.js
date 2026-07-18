const form = document.querySelector("#adviceForm");
const imageInput = document.querySelector("#imageInput");
const previewImage = document.querySelector("#previewImage");
const photoDrop = document.querySelector("#photoDrop");
const problemText = document.querySelector("#problemText");
const problemChips = document.querySelector("#problemChips");
const resultCard = document.querySelector("#resultCard");
const progressCard = document.querySelector("#progressCard");
const submitButton = document.querySelector("#submitButton");
const locationButton = document.querySelector("#locationButton");
const latInput = document.querySelector("#latInput");
const lonInput = document.querySelector("#lonInput");
const installButton = document.querySelector("#installButton");

let installPrompt = null;
let latestAdvice = null;

const escapeHtml = (value = "") =>
  String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);

const listItems = (items = []) => {
  const rows = items.filter(Boolean).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return rows ? `<ol>${rows}</ol>` : "<p>No action available.</p>";
};

const riskClass = (risk = "unknown") => {
  const normalized = String(risk).toLowerCase();
  if (["critical", "high"].includes(normalized)) return "risk-high";
  if (normalized === "medium") return "risk-medium";
  if (normalized === "low") return "risk-low";
  return "risk-unknown";
};

function setProgress(active) {
  progressCard.hidden = !active;
  submitButton.disabled = active;
  submitButton.textContent = active ? "Checking crop..." : "Check Crop Health";
  document.querySelectorAll(".steps span").forEach((step) => step.classList.remove("done"));
  if (!active) return;

  const steps = [...document.querySelectorAll(".steps span")];
  steps.forEach((step, index) => {
    window.setTimeout(() => step.classList.add("done"), 450 + index * 650);
  });
}

function renderAdvice(advice) {
  latestAdvice = advice;
  const risk = String(advice?.risk_level || "unknown");
  const confidence = Number(advice?.confidence || 0);
  const confidenceText = confidence > 0
    ? `${Math.round(confidence * 100)}% (${escapeHtml(advice?.confidence_label || "unknown")})`
    : "Unknown";
  const technical = advice?.technical || {};
  const helpline = advice?.helpline || {};

  resultCard.className = `result-card ${riskClass(risk)}`;
  resultCard.innerHTML = `
    <div class="result-topline">
      <p class="eyebrow">Crop health result</p>
      <span class="risk-pill">${escapeHtml(risk.toUpperCase())} RISK</span>
    </div>
    <h2>${escapeHtml(advice?.problem || "Crop advice")}</h2>
    <p class="summary">${escapeHtml(advice?.summary || "Advice is ready.")}</p>

    <div class="metric-grid">
      <div class="metric"><span>Crop</span><strong>${escapeHtml(advice?.crop || "Unknown")}</strong></div>
      <div class="metric"><span>Status</span><strong>${escapeHtml(String(advice?.status || "ready").replaceAll("_", " "))}</strong></div>
      <div class="metric"><span>Confidence</span><strong>${confidenceText}</strong></div>
    </div>

    <section class="advice-section">
      <h3>What to do today</h3>
      ${listItems(advice?.what_to_do_today)}
    </section>

    <section class="advice-section">
      <h3>Treatment guidance</h3>
      <p>${escapeHtml(advice?.treatment_guidance || "Use only approved treatment after local confirmation.")}</p>
    </section>

    <section class="advice-section">
      <h3>Do not do this</h3>
      ${listItems(advice?.what_not_to_do)}
    </section>

    <section class="advice-section expert-card">
      <h3>When to call an expert</h3>
      <p>${escapeHtml(advice?.when_to_call_expert || "Call a local agriculture officer if symptoms spread.")}</p>
      <strong>${escapeHtml(helpline.name || "Kisan Call Center")}: ${escapeHtml(helpline.number || "1800-180-1551")}</strong>
    </section>

    <details>
      <summary>Advanced details</summary>
      <div class="metric-grid">
        <div class="metric"><span>Model</span><strong>${escapeHtml(technical.model_source || "unknown")}</strong></div>
        <div class="metric"><span>Label</span><strong>${escapeHtml(technical.disease_label || "unknown")}</strong></div>
        <div class="metric"><span>Yield risk</span><strong>${escapeHtml(technical.yield_loss_range || "varies")}</strong></div>
      </div>
    </details>

    <section class="advice-section feedback-card">
      <h3>Help improve AgriBloom</h3>
      <p>Was this advice useful for your crop?</p>
      <div class="feedback-actions">
        <button type="button" data-feedback="useful">Useful</button>
        <button type="button" data-feedback="not_useful">Not useful</button>
      </div>
      <textarea id="feedbackCorrection" rows="3" placeholder="Optional: tell us the correct crop, disease, or what happened in the field."></textarea>
      <p id="feedbackStatus" class="feedback-status"></p>
    </section>
  `;
}

function renderError(message) {
  latestAdvice = null;
  resultCard.className = "result-card error-card";
  resultCard.innerHTML = `
    <p class="eyebrow">Could not complete check</p>
    <h2>Please try again</h2>
    <p class="summary">${escapeHtml(message)}</p>
  `;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  previewImage.src = URL.createObjectURL(file);
  previewImage.hidden = false;
  photoDrop.classList.add("has-image");
});

resultCard.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-feedback]");
  if (!button || !latestAdvice) return;

  const status = document.querySelector("#feedbackStatus");
  const correction = document.querySelector("#feedbackCorrection")?.value || "";
  button.disabled = true;
  if (status) status.textContent = "Saving feedback...";

  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "web",
        rating: button.dataset.feedback,
        crop: latestAdvice.crop,
        problem: latestAdvice.problem,
        language: document.querySelector("#language")?.value || latestAdvice.language || "en",
        farmer_text: problemText.value,
        correction,
        advice: latestAdvice,
      }),
    });
    if (!response.ok) throw new Error("Feedback could not be saved.");
    if (status) status.textContent = "Feedback saved for review.";
  } catch (error) {
    if (status) status.textContent = error.message || "Feedback failed.";
  } finally {
    button.disabled = false;
  }
});

problemChips.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-problem]");
  if (!button) return;

  document.querySelectorAll("#problemChips button").forEach((chip) => chip.classList.remove("active"));
  button.classList.add("active");
  const current = problemText.value.trim();
  const problem = button.dataset.problem;
  problemText.value = current ? `${current}\n${problem}` : problem;
});

locationButton.addEventListener("click", () => {
  if (!navigator.geolocation) {
    renderError("Location is not available on this device. You can type state and district manually.");
    return;
  }

  locationButton.disabled = true;
  locationButton.textContent = "Finding location...";
  navigator.geolocation.getCurrentPosition(
    (position) => {
      latInput.value = position.coords.latitude;
      lonInput.value = position.coords.longitude;
      locationButton.textContent = "Location added";
      locationButton.disabled = false;
    },
    () => {
      renderError("Location permission was denied. You can still continue by typing state and district.");
      locationButton.textContent = "Use my location";
      locationButton.disabled = false;
    },
    { enableHighAccuracy: false, timeout: 7000, maximumAge: 600000 },
  );
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setProgress(true);

  try {
    const body = new FormData(form);
    const response = await fetch("/api/analyze", { method: "POST", body });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Crop analysis failed.");
    }
    if (!payload.farmer_advice) {
      throw new Error("The backend did not return farmer advice.");
    }

    localStorage.setItem("agribloom:lastAdvice", JSON.stringify(payload.farmer_advice));
    renderAdvice(payload.farmer_advice);
  } catch (error) {
    renderError(error.message || "Something went wrong.");
  } finally {
    setProgress(false);
  }
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  installButton.hidden = false;
});

installButton.addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  installButton.hidden = true;
});

try {
  const lastAdvice = JSON.parse(localStorage.getItem("agribloom:lastAdvice") || "null");
  if (lastAdvice) renderAdvice(lastAdvice);
} catch {
  localStorage.removeItem("agribloom:lastAdvice");
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/service-worker.js").catch(() => {});
}
