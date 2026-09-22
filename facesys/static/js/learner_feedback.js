// Learner Feedback form: clickable 5-star rating and per-item like/dislike toggles,
// both mirrored into hidden inputs so they submit with the rest of the plain <form>.

(function initStars() {
  const stars = document.querySelectorAll("#star-rating .star");
  const ratingInput = document.getElementById("rating-input");
  if (!stars.length) return;

  function paint(value) {
    stars.forEach((s) => s.classList.toggle("filled", Number(s.dataset.value) <= value));
  }

  stars.forEach((star) => {
    star.addEventListener("click", () => {
      const value = Number(star.dataset.value);
      ratingInput.value = value;
      paint(value);
    });
    star.addEventListener("mouseenter", () => paint(Number(star.dataset.value)));
  });
  const ratingWrap = document.getElementById("star-rating");
  ratingWrap.addEventListener("mouseleave", () => paint(Number(ratingInput.value)));
})();

(function initFlags() {
  document.querySelectorAll(".flag-row").forEach((row) => {
    const key = row.dataset.key;
    const hidden = document.getElementById("flag_" + key);
    const likeBtn = row.querySelector(".flag-btn.like");
    const dislikeBtn = row.querySelector(".flag-btn.dislike");

    function setState(value) {
      hidden.value = value || "";
      likeBtn.classList.toggle("active", value === "like");
      dislikeBtn.classList.toggle("active", value === "dislike");
    }

    likeBtn.addEventListener("click", () => setState(hidden.value === "like" ? "" : "like"));
    dislikeBtn.addEventListener("click", () => setState(hidden.value === "dislike" ? "" : "dislike"));
  });
})();

document.getElementById("feedback-form")?.addEventListener("submit", (e) => {
  const rating = Number(document.getElementById("rating-input").value);
  if (!rating) {
    e.preventDefault();
    alert("Please select a star rating before submitting.");
  }
});
