/* T58 shared page chrome.
   Loaded once by _sidebar.html (included on every page except the bare
   "_job" progress pages), so this runs on every screen without having to
   touch each template individually.

   Responsibilities:
   - build + inject the sticky topbar (breadcrumb, engine status, theme toggle)
   - persist + apply the light/dark theme choice
   - remember which sidebar groups the person had open/closed
   - a small animateNumber() helper other pages can call for counted-up KPIs
*/
(function () {
  "use strict";

  var THEME_KEY = "t58-theme";
  var GROUP_KEY_PREFIX = "t58-navgroup:";

  function applyStoredTheme() {
    var saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
    if (saved === "light") document.documentElement.setAttribute("data-theme", "light");
  }
  // Apply before paint where possible to avoid a flash of the wrong theme.
  applyStoredTheme();

  function toggleTheme() {
    var isLight = document.documentElement.getAttribute("data-theme") === "light";
    var next = isLight ? "dark" : "light";
    if (next === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
    var btn = document.getElementById("t58-theme-toggle");
    if (btn) btn.textContent = next === "light" ? "\u263D" : "\u2600";
  }

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

  function pageBreadcrumb() {
    var activeItem = document.querySelector(".t58-nav-item.active");
    var section = "";
    if (activeItem) {
      var group = activeItem.closest(".t58-nav-group");
      if (group) {
        var summary = group.querySelector("summary");
        if (summary) section = summary.textContent.trim().replace(/[\u25B6\u25B8]/g, "").trim();
      }
    }
    var pageTitleEl = document.querySelector(".t58-page-header h1");
    var pageTitle = pageTitleEl ? pageTitleEl.textContent.trim() : document.title;
    return section ? (section + " / " + pageTitle) : pageTitle;
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
    var topbar = document.querySelector(".t58-topbar");
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

    if (topbar && topbar.nextSibling) main.insertBefore(bar, topbar.nextSibling);
    else main.insertBefore(bar, main.firstChild);
  }

  function buildTopbar() {
    var main = document.querySelector(".t58-main");
    if (!main || document.querySelector(".t58-topbar")) return;

    var bar = document.createElement("div");
    bar.className = "t58-topbar";

    var crumb = document.createElement("div");
    crumb.className = "crumb";
    crumb.innerHTML = "T58 &nbsp;/&nbsp; <b>" + pageBreadcrumb() + "</b>";

    var right = document.createElement("div");
    right.className = "right";

    var status = document.createElement("div");
    status.className = "t58-engine-status";
    status.id = "t58-engine-status";
    status.innerHTML = '<span class="dot"></span><span class="txt">Checking&hellip;</span>';

    var themeBtn = document.createElement("button");
    themeBtn.className = "t58-theme-toggle";
    themeBtn.id = "t58-theme-toggle";
    themeBtn.type = "button";
    themeBtn.setAttribute("aria-label", "Toggle light / dark theme");
    themeBtn.textContent = document.documentElement.getAttribute("data-theme") === "light" ? "\u263D" : "\u2600";
    themeBtn.addEventListener("click", toggleTheme);

    right.appendChild(status);
    right.appendChild(themeBtn);
    bar.appendChild(crumb);
    bar.appendChild(right);
    main.insertBefore(bar, main.firstChild);

    // Best-effort heartbeat -- reuses the dashboard's existing JSON feed
    // purely as a "the Flask process is alive" ping. Never blocks the UI.
    function ping() {
      fetch("/api/dashboard-data", { cache: "no-store" })
        .then(function (r) {
          status.className = "t58-engine-status " + (r.ok ? "online" : "offline");
          status.querySelector(".txt").textContent = r.ok ? "Engine online" : "Engine unreachable";
        })
        .catch(function () {
          status.className = "t58-engine-status offline";
          status.querySelector(".txt").textContent = "Engine unreachable";
        });
    }
    ping();
    setInterval(ping, 20000);
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
    buildTopbar();
    buildStepper();
    restoreNavGroups();
    animateNumbers();
  });
})();
