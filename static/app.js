document.addEventListener("DOMContentLoaded", () => {
  const slotsContainer = document.getElementById("drugSlotsContainer");
  const addDrugBtn = document.getElementById("addDrugBtn");
  const predictBtn = document.getElementById("predictBtn");
  const resultsContainer = document.getElementById("resultsContainer");

  let allDrugs = [];
  let currentSlotCount = 0;

  // Fetch available drugs
  fetch("/api/drugs")
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        allDrugs = data.drugs;
        // Default sample: Warfarin
        addDrugSlot(allDrugs.find(d => d.name.toLowerCase().includes("warfarin"))?.name || "Warfarin");
      }
    })
    .catch(err => console.error("Error fetching drug list:", err));

  function addDrugSlot(initialVal = "") {
    if (currentSlotCount >= 4) {
      alert("Maximum 4 medications can be analyzed concurrently.");
      return;
    }

    currentSlotCount++;
    const slotId = `drugInput_${currentSlotCount}`;
    const listId = `drugList_${currentSlotCount}`;

    const slotDiv = document.createElement("div");
    slotDiv.className = "slot-row";
    slotDiv.dataset.slotId = currentSlotCount;

    slotDiv.innerHTML = `
      <div class="select-wrapper">
        <input type="text" id="${slotId}" placeholder="Medication ${currentSlotCount} (e.g. Warfarin, Aspirin, Metformin...)" autocomplete="off" value="${initialVal}" />
        <div id="${listId}" class="dropdown-list"></div>
      </div>
      ${currentSlotCount > 1 ? `<button type="button" class="btn-slot-remove" data-remove="${currentSlotCount}">Remove</button>` : ''}
    `;

    slotsContainer.appendChild(slotDiv);

    const input = document.getElementById(slotId);
    const list = document.getElementById(listId);
    setupAutocomplete(input, list);

    const removeBtn = slotDiv.querySelector(".btn-slot-remove");
    if (removeBtn) {
      removeBtn.addEventListener("click", () => {
        slotDiv.remove();
        currentSlotCount--;
      });
    }
  }

  addDrugBtn.addEventListener("click", () => addDrugSlot(""));

  function setupAutocomplete(input, list) {
    input.addEventListener("input", () => {
      const val = input.value.trim().toLowerCase();
      if (!val) {
        list.style.display = "none";
        return;
      }

      const matches = allDrugs.filter(d => 
        d.name.toLowerCase().includes(val) || d.cid.toLowerCase().includes(val)
      ).slice(0, 10);

      if (matches.length === 0) {
        list.style.display = "none";
        return;
      }

      list.innerHTML = matches.map(d => 
        `<div class="dropdown-item" data-value="${d.name}">${d.display}</div>`
      ).join("");
      list.style.display = "block";
    });

    list.addEventListener("click", (e) => {
      if (e.target.classList.contains("dropdown-item")) {
        input.value = e.target.dataset.value;
        list.style.display = "none";
      }
    });

    document.addEventListener("click", (e) => {
      if (!input.contains(e.target) && !list.contains(e.target)) {
        list.style.display = "none";
      }
    });
  }

  predictBtn.addEventListener("click", () => {
    const inputs = slotsContainer.querySelectorAll("input");
    const drugs = Array.from(inputs).map(i => i.value.trim()).filter(v => v.length > 0);

    if (drugs.length === 0) {
      alert("Please enter at least 1 medication name.");
      return;
    }

    predictBtn.disabled = true;
    predictBtn.querySelector(".btn-text").textContent = "Analyzing Clinical Data...";

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drugs })
    })
      .then(res => res.json())
      .then(data => {
        predictBtn.disabled = false;
        predictBtn.querySelector(".btn-text").textContent = "Analyze Regimen Safety";

        if (!data.success) {
          alert(data.error || "Prediction failed.");
          return;
        }

        renderResults(data.result);
      })
      .catch(err => {
        predictBtn.disabled = false;
        predictBtn.querySelector(".btn-text").textContent = "Analyze Regimen Safety";
        alert("Error connecting to server.");
        console.error(err);
      });
  });

  function renderResults(res) {
    resultsContainer.classList.remove("hidden");
    resultsContainer.scrollIntoView({ behavior: "smooth" });

    // Render FDA Black Box Alert
    const blackboxAlert = document.getElementById("blackboxAlert");
    if (res.has_blackbox && res.blackbox_warning) {
      blackboxAlert.classList.remove("hidden");
      document.getElementById("blackboxText").textContent = res.blackbox_warning;
    } else {
      blackboxAlert.classList.add("hidden");
    }

    // Render Regimen Tags
    const tagsContainer = document.getElementById("resRegimenTags");
    tagsContainer.innerHTML = res.drugs.map(d => `<span class="drug-tag">${d.display}</span>`).join('<span class="plus">+</span>');

    document.getElementById("resRiskLevel").textContent = res.risk_level;
    document.getElementById("resRiskLevel").style.color = res.risk_color;
    document.getElementById("riskScoreVal").textContent = `${res.overall_risk_score}%`;
    document.getElementById("resSummaryText").textContent = res.summary_text;

    // Animate Gauge SVG
    const gaugeFill = document.getElementById("gaugeFill");
    const strokeDash = 314;
    const offset = strokeDash - (strokeDash * (res.overall_risk_score / 100));
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = res.risk_color;

    // Render Driver Pair Alert (If > 1 Drug)
    const driverAlert = document.getElementById("driverPairAlert");
    const pairwiseCard = document.getElementById("pairwiseCard");

    if (res.num_drugs > 1 && res.driver_pair) {
      driverAlert.classList.remove("hidden");
      document.getElementById("driverPairTitle").textContent = res.driver_pair.pair_display;
      document.getElementById("driverScore").textContent = `${res.driver_pair.pair_risk_score}% Risk`;

      pairwiseCard.classList.remove("hidden");
      const pGrid = document.getElementById("pairwiseGrid");
      pGrid.innerHTML = res.pairwise_breakdown.map(p => `
        <div class="pair-card">
          <div class="pair-title">${p.pair_display}</div>
          <div class="pair-stat">Interaction Risk: <strong>${p.pair_risk_score}%</strong></div>
          <div class="pair-stat">Vector Similarity: ${p.spatial_cosine_similarity}</div>
        </div>
      `).join("");
    } else {
      driverAlert.classList.add("hidden");
      pairwiseCard.classList.add("hidden");
    }

    // Render Side Effects Grid
    const seTitle = document.getElementById("seSectionTitle");
    seTitle.textContent = res.num_drugs === 1 ? "Extracted Monotherapy Clinical Adverse Events" : "Synergistic Polypharmacy Side Effects";

    const sideEffects = res.num_drugs === 1 ? res.isolated_side_effects : res.top_predicted_side_effects;
    const grid = document.getElementById("sideEffectsGrid");

    grid.innerHTML = sideEffects.map(se => {
      const probText = se.probability ? ` (${se.probability}%)` : '';
      const probVal = se.probability || 50;
      return `
        <div class="se-card">
          <div class="se-header">
            <span class="se-name">${se.side_effect}</span>
            <span class="se-badge ${se.severity}">${se.severity}${probText}</span>
          </div>
          <div class="se-progress-bar">
            <div class="se-progress-fill" style="width: ${probVal}%;"></div>
          </div>
        </div>
      `;
    }).join("");
  }
});
