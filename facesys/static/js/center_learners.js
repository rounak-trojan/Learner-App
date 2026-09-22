// Center Learner Details page: the Active switch flips a learner's active flag via a
// quick POST, and Show More opens a modal with that learner's full details (atlas
// enrollment info, recent attendance, pivoted test results) fetched on demand.

document.querySelectorAll(".active-toggle").forEach((toggle) => {
  toggle.addEventListener("change", async () => {
    const luid = toggle.dataset.luid;
    const prev = !toggle.checked;
    try {
      const resp = await fetch(`/center/learners/${encodeURIComponent(luid)}/toggle-active`, { method: "POST" });
      if (!resp.ok) throw new Error("Request failed");
      const data = await resp.json();
      toggle.checked = data.active;
    } catch (err) {
      toggle.checked = prev; // revert on failure
      alert("Could not update active status. Try again.");
    }
  });
});

const overlay = document.getElementById("learner-modal-overlay");
const modalBody = document.getElementById("modal-body");
const closeBtn = document.getElementById("modal-close-btn");

function closeModal() {
  overlay.classList.remove("open");
}
closeBtn.addEventListener("click", closeModal);
overlay.addEventListener("click", (e) => {
  if (e.target === overlay) closeModal();
});

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

document.querySelectorAll(".show-more-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const luid = btn.dataset.luid;
    modalBody.innerHTML = "Loading…";
    overlay.classList.add("open");
    try {
      const resp = await fetch(`/center/learners/${encodeURIComponent(luid)}/full`);
      if (!resp.ok) throw new Error("Request failed");
      const data = await resp.json();
      renderModal(data);
    } catch (err) {
      modalBody.innerHTML = "Could not load learner details.";
    }
  });
});

function renderModal(data) {
  const l = data.learner;
  const atlas = data.atlas;

  let html = `<h2 style="margin-top:0;">${escapeHtml(l.name)}</h2>`;
  html += `<p class="hint" style="margin-bottom:0;">LUID ${escapeHtml(l.luid)} · Roll No ${escapeHtml(l.roll_no)} · ${l.active ? "Active" : "Inactive"}</p>`;

  html += `<h3>Enrollment</h3>`;
  if (atlas) {
    html += `<div class="table-scroll"><table class="data-table"><tbody>
      <tr><th>Batch</th><td>${escapeHtml(atlas.batch_name)}</td></tr>
      <tr><th>Goal</th><td>${escapeHtml(atlas.goal_name)}</td></tr>
      <tr><th>Centre</th><td>${escapeHtml(atlas.centre_name)}</td></tr>
      <tr><th>City</th><td>${escapeHtml(atlas.centre_city_name)}</td></tr>
    </tbody></table></div>`;
  } else {
    html += `<p class="hint">No Atlas enrollment data found for this learner (subscription may have expired).</p>`;
  }

  html += `<h3>Recent Attendance</h3>`;
  html += `<a class="btn btn-sm btn-secondary" href="/center/learners/${encodeURIComponent(l.luid)}/attendance">View Full Attendance Calendar</a>`;
  if (data.attendance.length) {
    html += `<div class="table-scroll"><table class="data-table"><thead><tr><th>Time</th><th>Type</th><th>Status</th></tr></thead><tbody>`;
    for (const a of data.attendance) {
      html += `<tr><td>${escapeHtml(a.ts)}</td><td>${escapeHtml(a.punch_type)}</td><td>${escapeHtml(a.status)}</td></tr>`;
    }
    html += `</tbody></table></div>`;
  } else {
    html += `<p class="hint">No attendance recorded yet.</p>`;
  }

  html += `<h3>Test Results</h3>`;
  if (data.tests.length) {
    html += `<div class="table-scroll"><table class="data-table"><thead><tr><th>Test</th><th>Date</th><th>Sections</th><th>Total</th></tr></thead><tbody>`;
    for (const t of data.tests) {
      const sections = t.sections.map((s) => `${escapeHtml(s.name)}: ${escapeHtml(s.score)}`).join(", ");
      html += `<tr><td>${escapeHtml(t.test_title)}</td><td>${escapeHtml(t.test_start_date)}</td><td>${sections}</td><td>${t.total}</td></tr>`;
    }
    html += `</tbody></table></div>`;
  } else {
    html += `<p class="hint">No test results yet.</p>`;
  }

  modalBody.innerHTML = html;
}
