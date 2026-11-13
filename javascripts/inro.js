// The landing page's menu takes its background once the page scrolls.
const followScroll = () =>
  document.documentElement.classList.toggle("inro-scrolled", window.scrollY > 8);
window.addEventListener("scroll", followScroll, { passive: true });
followScroll();

// A link to the page it is on scrolls to the top. Instant navigation would revert the click once
// the page has scrolled, since scrolling rewrites the URL's hash behind its back.
document.addEventListener(
  "click",
  (event) => {
    const link = event.target.closest("a[href]");
    if (!link || link.target || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const [page, hash] = link.href.split("#");
    if (page !== location.href.split("#")[0] || hash) return;
    event.preventDefault();
    event.stopPropagation();
    document.getElementById("__drawer").checked = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  },
  true,
);

// Runs on every page, including those reached through instant navigation.
document$.subscribe(() => {
  // FUTON's rows stand out in the tables, and results tables keep their numbers on one line.
  document.querySelectorAll(".md-typeset table:not([class])").forEach((table) => {
    table.querySelectorAll("tbody tr").forEach((row) => {
      if (/^FUTON\b/.test(row.cells[0]?.textContent.trim() ?? "")) row.classList.add("inro-futon");
    });
    if (/[\u2191\u2193]/.test(table.tHead?.textContent ?? "")) {
      table.closest(".md-typeset__table")?.classList.add("inro-results");
    }
  });

  // Before-and-after comparisons follow their slider.
  document.querySelectorAll("[data-compare]").forEach((figure) => {
    const slider = figure.querySelector("input[type=range]");
    const follow = () => figure.style.setProperty("--split", `${slider.value}%`);
    slider.addEventListener("input", follow);
    follow();
  });
});
