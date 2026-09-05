/* T58 shared page chrome.
   Loaded once by _sidebar.html (included on every page except the bare
   "_job" progress pages), so this runs on every screen without having to
   touch each template individually.

   Responsibilities:
   - build + inject the persistent stage stepper
   - remember which sidebar groups the person had open/closed
   - a small animateNumber() helper other pages can call for counted-up KPIs

   (The old sticky top bar -- breadcrumb + engine-status pill + light/dark
   toggle -- was removed per direct feedback that it read as an unwanted
   banner across the top of every page. The theme toggle isn't currently
   exposed anywhere else; reintroduce it as a small icon inside the
   sidebar itself if a light/dark switch is wanted back.)
*/
(function () {
  "use strict";

  var GROUP_KEY_PREFIX = "t58-navgroup:";

  function restoreNavGroups() {
    document.querySelectorAll(".t58-nav-group").forEach(function (group, idx) {
      var summary = group.querySelector("summary");
      var key = GROUP_KEY_PREFIX + (summary ? summary.textContent.trim() : idx);
      var saved = null;
      try { saved = localStorage.getItem(key); } catch (e) {}
      // Only override the server-rendered default (open on the active
      // section) if the person has explicitly toggled this group before.
      if (saved === "open") group.setAttribute("open", "");
      else if (saved === "closed") group.removeAttribute("open");
      group.addEventListener("toggle", function () {
        try { localStorage.setItem(key, group.open ? "open" : "closed"); } catch (e) {}
      });
    });
  }

  /* The 8-stage journey: Create -> Test -> Optimize -> Validate ->
     Champion -> Forward Test -> Deploy -> Monitor. Keyed by URL path so it
     works with zero coupling to how each template names active_page. This
     is a navigation aid only -- it marks which stage the current page
     belongs to, not whether that stage is "done" for any given strategy
     (the app has no persisted "current strategy" to make that claim about
     honestly). */
  var STAGES = [
    { name: "Create", href: "/speed-run", paths: ["/speed-run", "/generate-strategies", "/research-agent"] },
    { name: "Test", href: "/", paths: ["/", "/payout-probability"] },
    { name: "Optimize", href: "/search", paths: ["/search", "/refine", "/full-pipeline", "/quick-optimize", "/multi-objective", "/evolution"] },
    { name: "Validate", href: "/walk-forward-opt", paths: ["/walk-forward-opt", "/walk-forward-ga", "/cpcv", "/sensitivity", "/regime-matrix"] },
    { name: "Champion", href: "/portfolio", paths: ["/portfolio", "/ensemble", "/family-diversity"] },
    { name: "Forward Test", href: "/forward-test", paths: ["/forward-test"] },
    { name: "Deploy", href: "/deploy-live", paths: ["/deploy-live"] },
    { name: "Monitor", href: "/live-market", paths: ["/live-market"] }
  ];

  function currentStageIndex() {
    var path = window.location.pathname.replace(/\/+$/, "") || "/";
    for (var i = 0; i < STAGES.length; i++) {
      if (STAGES[i].paths.indexOf(path) !== -1) return i;
    }
    return -1;
  }

  function buildStepper() {
    var main = document.querySelector(".t58-main");
    if (!main || document.querySelector(".t58-stepper")) return;

    var current = currentStageIndex();
    var bar = document.createElement("div");
    bar.className = "t58-stepper";
    STAGES.forEach(function (stage, i) {
      var a = document.createElement("a");
      a.href = stage.href;
      a.textContent = stage.name;
      if (i === current) a.className = "current";
      bar.appendChild(a);
      if (i < STAGES.length - 1) {
        var sep = document.createElement("span");
        sep.className = "sep";
        sep.textContent = "\u2192";
        bar.appendChild(sep);
      }
    });

    main.insertBefore(bar, main.firstChild);
  }

  /* Simple counted-up number animation for hero KPIs. Any element with
     [data-animate-to] gets counted from 0 (or its current text) up to the
     target on load. Safe no-op if nothing on the page uses it. */
  function animateNumbers() {
    document.querySelectorAll("[data-animate-to]").forEach(function (el) {
      var target = parseFloat(el.getAttribute("data-animate-to"));
      var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      var suffix = el.getAttribute("data-suffix") || "";
      if (isNaN(target)) return;
      var start = 0;
      var duration = 700;
      var startTime = null;
      function step(ts) {
        if (startTime === null) startTime = ts;
        var progress = Math.min(1, (ts - startTime) / duration);
        var eased = 1 - Math.pow(1 - progress, 3);
        var val = start + (target - start) * eased;
        el.textContent = val.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

  window.T58Chrome = { animateNumbers: animateNumbers };

  document.addEventListener("DOMContentLoaded", function () {
    buildStepper();
    restoreNavGroups();
    animateNumbers();
  });
})();
