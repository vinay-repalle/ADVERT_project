(() => {
  function initAdvretProgress() {
    const form = document.getElementById("advret-form");
    const runBtn = document.getElementById("run-btn");
    const progressModal = document.getElementById("simple-progress-modal");
    const progressBar = document.getElementById("simple-progress-bar");
    const progressPct = document.getElementById("simple-progress-pct");
    const progressTrack = document.getElementById("simple-progress-track");

    if (!form || !runBtn || !progressModal || !progressBar || !progressPct || !progressTrack) return;

    let submitting = false;
    let timer = null;

    function setProgress(pct) {
      const clamped = Math.max(0, Math.min(100, pct));
      progressBar.style.width = clamped + "%";
      progressPct.textContent = Math.round(clamped) + "%";
      progressTrack.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }

    function startApproximateProgress() {
      const start = Date.now();
      setProgress(1);
      timer = window.setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        // No real progress tracking (no polling). This is a smooth, time-based
        // approximation that continues to animate while the request runs.
        // Ease out toward 100%, reaching 100% after ~18 seconds.
        const eased = 100 * (1 - Math.exp(-elapsed / 3.2));
        const capped = elapsed >= 18 ? 100 : Math.min(99, eased);
        setProgress(capped);
      }, 200);
    }

    form.addEventListener("submit", (e) => {
      if (submitting) return;
      submitting = true;

      // Let the modal paint before the browser starts uploading / navigating.
      e.preventDefault();

      runBtn.disabled = true;
      runBtn.classList.add("opacity-70", "cursor-not-allowed");
      progressModal.classList.remove("hidden");
      startApproximateProgress();

      // Two rAFs gives the browser a reliable chance to render the modal.
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
          form.submit();
        });
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAdvretProgress);
  } else {
    initAdvretProgress();
  }
})();

