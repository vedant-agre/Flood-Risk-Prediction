// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Update slider value labels
function updateVal(id) {
    const val = document.getElementById(id).value;
    document.getElementById('val-' + id).innerText = val;
}

// Preset Configurations
const PRESETS = {
    baseline: {
        MonsoonIntensity: 3, TopographyDrainage: 8, RiverManagement: 8, Deforestation: 3,
        Urbanization: 3, ClimateChange: 3, DamsQuality: 8, Siltation: 2,
        AgriculturalPractices: 3, Encroachments: 2, IneffectiveDisasterPreparedness: 2,
        DrainageSystems: 8, CoastalVulnerability: 3, Landslides: 2, Watersheds: 8,
        DeterioratingInfrastructure: 2, PopulationScore: 3, WetlandLoss: 2,
        InadequatePlanning: 3, PoliticalFactors: 3
    },
    monsoon: {
        MonsoonIntensity: 12, TopographyDrainage: 5, RiverManagement: 5, Deforestation: 7,
        Urbanization: 7, ClimateChange: 9, DamsQuality: 5, Siltation: 8,
        AgriculturalPractices: 6, Encroachments: 7, IneffectiveDisasterPreparedness: 6,
        DrainageSystems: 4, CoastalVulnerability: 7, Landslides: 8, Watersheds: 4,
        DeterioratingInfrastructure: 7, PopulationScore: 8, WetlandLoss: 6,
        InadequatePlanning: 7, PoliticalFactors: 5
    },
    catastrophic: {
        MonsoonIntensity: 14, TopographyDrainage: 2, RiverManagement: 2, Deforestation: 12,
        Urbanization: 12, ClimateChange: 13, DamsQuality: 2, Siltation: 12,
        AgriculturalPractices: 9, Encroachments: 12, IneffectiveDisasterPreparedness: 12,
        DrainageSystems: 1, CoastalVulnerability: 11, Landslides: 11, Watersheds: 2,
        DeterioratingInfrastructure: 13, PopulationScore: 12, WetlandLoss: 12,
        InadequatePlanning: 12, PoliticalFactors: 11
    },
    urban: {
        MonsoonIntensity: 8, TopographyDrainage: 3, RiverManagement: 4, Deforestation: 6,
        Urbanization: 13, ClimateChange: 6, DamsQuality: 5, Siltation: 10,
        AgriculturalPractices: 4, Encroachments: 12, IneffectiveDisasterPreparedness: 8,
        DrainageSystems: 2, CoastalVulnerability: 6, Landslides: 3, Watersheds: 3,
        DeterioratingInfrastructure: 9, PopulationScore: 13, WetlandLoss: 10,
        InadequatePlanning: 11, PoliticalFactors: 7
    }
};

function loadPreset(presetKey) {
    const config = PRESETS[presetKey];
    if (!config) return;
    
    for (const [key, val] of Object.entries(config)) {
        const slider = document.getElementById(key);
        if (slider) {
            slider.value = val;
            updateVal(key);
        }
    }
    runAssessment();
}

// Run Assessment via API Call
async function runAssessment() {
    const inputs = {};
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        inputs[slider.name] = parseFloat(slider.value);
    });

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });

        const res = await response.json();
        if (res.status === 'success') {
            const pred = res.prediction;

            // 1. Update Tier Badge
            const badge = document.getElementById('risk-tier-badge');
            badge.innerText = `${pred.risk_tier.toUpperCase()} RISK`;
            badge.className = 'risk-badge';
            if (pred.risk_tier === 'Low') badge.classList.add('badge-low');
            else if (pred.risk_tier === 'Moderate') badge.classList.add('badge-moderate');
            else if (pred.risk_tier === 'High') badge.classList.add('badge-high');
            else if (pred.risk_tier === 'Severe') badge.classList.add('badge-severe');

            // 2. Update Probability Score & Meter Bar
            const probPct = pred.flood_probability_score;
            document.getElementById('prob-percentage').innerText = `${probPct.toFixed(1)}%`;
            document.getElementById('meter-bar').style.width = `${Math.min(100, Math.max(5, probPct))}%`;

            // 3. Update Confidence Breakdown Bars
            const probs = pred.probabilities;
            updateTierBar('low', probs['Low'] || 0);
            updateTierBar('mod', probs['Moderate'] || 0);
            updateTierBar('high', probs['High'] || 0);
            updateTierBar('sev', probs['Severe'] || 0);

            // 4. Update Mitigation Actions
            const mitList = document.getElementById('mitigation-list');
            mitList.innerHTML = '';
            pred.recommendations.forEach(action => {
                const li = document.createElement('li');
                li.innerHTML = formatMarkdown(action);
                mitList.appendChild(li);
            });
        }
    } catch (err) {
        console.error('Error executing risk assessment:', err);
    }
}

function updateTierBar(tierKey, value) {
    const valRounded = Math.round(value);
    document.getElementById(`prob-bar-${tierKey}`).style.width = `${valRounded}%`;
    document.getElementById(`prob-val-${tierKey}`).innerText = `${valRounded}%`;
}

function formatMarkdown(text) {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// Initial assessment on load
window.addEventListener('DOMContentLoaded', () => {
    runAssessment();
});
